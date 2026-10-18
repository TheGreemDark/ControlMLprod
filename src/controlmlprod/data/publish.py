import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path

from botocore.exceptions import ClientError

from controlmlprod.data.config import DATA_DIR, S3_BUCKET
from controlmlprod.data.s3_client import create_s3_client

LOGGER = logging.getLogger(__name__)

OUTPUT_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = OUTPUT_DIR / "annotations"

DATASET_NAME = "cifar10"
DATASET_VERSION = os.getenv("DATASET_VERSION", "v1")
S3_PREFIX = f"processed/{DATASET_NAME}/{DATASET_VERSION}"


def calculate_sha256(path: Path) -> str:
    """Calculate a file's SHA-256 checksum."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_git_commit() -> str:
    """Return the current Git commit, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[3],
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

    return result.stdout.strip()


def list_existing_objects(s3: object, prefix: str) -> list[str]:
    """List object keys under a prefix."""
    keys: list[str] = []
    continuation_token: str | None = None

    while True:
        parameters: dict[str, object] = {
            "Bucket": S3_BUCKET,
            "Prefix": f"{prefix}/",
        }

        if continuation_token is not None:
            parameters["ContinuationToken"] = continuation_token

        response = s3.list_objects_v2(**parameters)  # type: ignore[attr-defined]

        contents = response.get("Contents", [])
        if isinstance(contents, list):
            for item in contents:
                if isinstance(item, dict) and isinstance(item.get("Key"), str):
                    keys.append(item["Key"])

        if not response.get("IsTruncated", False):
            break

        token = response.get("NextContinuationToken")
        if not isinstance(token, str):
            raise RuntimeError(  # noqa: TRY004
                "S3 pagination response has no continuation token"
            )

        continuation_token = token

    return keys


def load_annotations(path: Path) -> list[dict[str, object]]:
    """Read annotation JSON and validate its top-level structure."""
    with path.open(encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise TypeError(f"Expected a JSON list in {path}")

    validated: list[dict[str, object]] = []

    for record in records:
        if not isinstance(record, dict):
            raise TypeError(f"Invalid annotation record in {path}")
        validated.append(record)

    return validated


def publish_dataset() -> None:
    """Publish the processed dataset without overwriting an existing version."""
    if not DATASET_VERSION.startswith("v") or not DATASET_VERSION[1:].isdigit():
        raise ValueError("DATASET_VERSION must look like 'v1', 'v2', and so on.")

    required_files = [
        ANNOTATIONS_DIR / "train.json",
        ANNOTATIONS_DIR / "val.json",
        ANNOTATIONS_DIR / "test.json",
        OUTPUT_DIR / "stats.json",
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required processed file is missing: {path}. Run the data pipeline first."
            )

    train_records = load_annotations(ANNOTATIONS_DIR / "train.json")
    val_records = load_annotations(ANNOTATIONS_DIR / "val.json")
    test_records = load_annotations(ANNOTATIONS_DIR / "test.json")

    s3 = create_s3_client()
    existing_keys = list_existing_objects(s3, S3_PREFIX)

    if existing_keys:
        raise FileExistsError(
            f"Dataset version already exists at s3://{S3_BUCKET}/{S3_PREFIX}/. "
            "Choose a new DATASET_VERSION, for example v2. "
            "Existing versions are never overwritten."
        )

    annotation_metadata: dict[str, dict[str, object]] = {}

    for split_name in ("train", "val", "test"):
        path = ANNOTATIONS_DIR / f"{split_name}.json"
        annotation_metadata[f"{split_name}.json"] = {
            "size_bytes": path.stat().st_size,
            "sha256": calculate_sha256(path),
            "records": len(
                {
                    "train": train_records,
                    "val": val_records,
                    "test": test_records,
                }[split_name]
            ),
        }

    stats_path = OUTPUT_DIR / "stats.json"

    manifest: dict[str, object] = {
        "dataset": DATASET_NAME,
        "version": DATASET_VERSION,
        "source": "CIFAR-10 Python archive",
        "counts": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
            "total": len(train_records) + len(val_records) + len(test_records),
        },
        "parameters": {
            "val_size": len(val_records),
            "split_seed": int(os.getenv("SPLIT_SEED", "42")),
        },
        "git_commit": get_git_commit(),
        "annotations": annotation_metadata,
        "statistics": {
            "file": "stats.json",
            "size_bytes": stats_path.stat().st_size,
            "sha256": calculate_sha256(stats_path),
        },
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    files_to_upload = [
        path
        for path in OUTPUT_DIR.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and path != ANNOTATIONS_DIR / "train_all.json"
    ]
    files_to_upload.append(manifest_path)

    LOGGER.info(
        "Publishing %d files to s3://%s/%s/",
        len(files_to_upload),
        S3_BUCKET,
        S3_PREFIX,
    )

    for index, path in enumerate(files_to_upload, start=1):
        relative_path = path.relative_to(OUTPUT_DIR).as_posix()
        key = f"{S3_PREFIX}/{relative_path}"

        LOGGER.info(
            "[%d/%d] Uploading %s",
            index,
            len(files_to_upload),
            relative_path,
        )

        try:
            s3.upload_file(str(path), S3_BUCKET, key)
        except ClientError:
            LOGGER.exception("Upload failed for %s", key)
            raise

    LOGGER.info(
        "Dataset published successfully: s3://%s/%s/",
        S3_BUCKET,
        S3_PREFIX,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    publish_dataset()


if __name__ == "__main__":
    main()
