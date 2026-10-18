import hashlib
import logging
from pathlib import Path

from botocore.exceptions import ClientError

from controlmlprod.data.config import RAW_DIR, S3_BUCKET
from controlmlprod.data.s3_client import create_s3_client

LOGGER = logging.getLogger(__name__)

S3_KEY = "raw/cifar-10-python.tar.gz"
ARCHIVE_NAME = "cifar-10-python.tar.gz"
EXPECTED_MD5 = "c58f30108f718f92721af3b95e74349a"


def calculate_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_raw_archive() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    destination = RAW_DIR / ARCHIVE_NAME

    if destination.exists():
        if calculate_md5(destination) == EXPECTED_MD5:
            LOGGER.info("Using verified local archive: %s", destination)
            return destination

        LOGGER.warning("Local archive checksum mismatch; downloading again")
        destination.unlink()

    s3 = create_s3_client()

    try:
        s3.head_object(Bucket=S3_BUCKET, Key=S3_KEY)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
            raise FileNotFoundError(
                f"Source archive is missing: s3://{S3_BUCKET}/{S3_KEY}. Run 'make ingest' first."
            ) from exc
        raise

    temporary = destination.with_suffix(".download")
    temporary.unlink(missing_ok=True)

    try:
        LOGGER.info("Downloading archive from S3: s3://%s/%s", S3_BUCKET, S3_KEY)
        s3.download_file(S3_BUCKET, S3_KEY, str(temporary))

        actual_md5 = calculate_md5(temporary)
        if actual_md5 != EXPECTED_MD5:
            raise ValueError(
                f"Archive checksum mismatch: expected {EXPECTED_MD5}, received {actual_md5}"
            )

        temporary.replace(destination)
        LOGGER.info("Archive downloaded and checksum verified")
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    download_raw_archive()


if __name__ == "__main__":
    main()
