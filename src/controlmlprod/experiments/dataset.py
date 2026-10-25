"""Download and load versioned CIFAR-10 datasets from S3."""

import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from math import isfinite
from pathlib import Path, PurePosixPath

from botocore.client import BaseClient
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor

from controlmlprod.data.config import DATA_DIR, S3_BUCKET
from controlmlprod.data.s3_client import create_s3_client

LOGGER = logging.getLogger(__name__)

CACHE_ROOT = DATA_DIR / "cache"
CACHE_MARKER = ".download-complete"
DOWNLOAD_WORKERS = 16
NUM_CLASSES = 10
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ImageRecord:
    """Image path and class label."""

    file_name: str
    label: int


def _normalize_s3_prefix(s3_path: str) -> str:
    """Validate and normalize a relative S3 prefix."""
    prefix = s3_path.rstrip("/")
    parts = prefix.split("/")

    if (
        not prefix
        or prefix.startswith("/")
        or "\\" in prefix
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"Invalid S3 dataset path: {s3_path!r}")

    return PurePosixPath(*parts).as_posix()


def _safe_relative_path(file_name: str) -> PurePosixPath:
    """Reject absolute paths and path traversal in dataset metadata."""
    parts = file_name.split("/")

    if (
        not file_name
        or file_name.startswith("/")
        or "\\" in file_name
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"Invalid relative image path: {file_name!r}")

    return PurePosixPath(*parts)


def _load_records(path: Path) -> list[ImageRecord]:
    """Load and validate annotation records from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, list):
        raise TypeError(f"Expected a JSON list in {path}")

    records: list[ImageRecord] = []

    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise TypeError(f"Invalid annotation at index {index} in {path}")

        file_name = item.get("file_name")
        label = item.get("label")

        if not isinstance(file_name, str):
            raise TypeError(f"Missing or invalid file_name at index {index} in {path}")

        _safe_relative_path(file_name)

        if not isinstance(label, int) or isinstance(label, bool) or not 0 <= label < NUM_CLASSES:
            raise ValueError(f"Invalid class label at index {index} in {path}")

        records.append(ImageRecord(file_name=file_name, label=label))

    if not records:
        raise ValueError(f"No annotation records found in {path}")

    return records


def load_normalization_stats(path: Path) -> tuple[list[float], list[float]]:
    """Load RGB mean and standard deviation from stats.json."""
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")

    def read_triplet(name: str) -> list[float]:
        values = payload.get(name)

        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"Expected three RGB values for {name!r} in {path}")

        result: list[float] = []

        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Invalid {name!r} value in {path}")

            number = float(value)
            if not isfinite(number):
                raise ValueError(f"Non-finite {name!r} value in {path}")

            result.append(number)

        return result

    mean = read_triplet("mean")
    std = read_triplet("std")

    if any(value <= 0 for value in std):
        raise ValueError(f"Standard deviations must be positive in {path}")

    return mean, std


def _validate_dataset_content(dataset_dir: Path) -> None:
    """Validate required metadata and all image paths referenced by annotations."""
    required_files = [
        dataset_dir / "manifest.json",
        dataset_dir / "stats.json",
        *(dataset_dir / "annotations" / f"{split}.json" for split in SPLITS),
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset file is missing: {path}")

    load_normalization_stats(dataset_dir / "stats.json")

    for split in SPLITS:
        annotation_path = dataset_dir / "annotations" / f"{split}.json"
        records = _load_records(annotation_path)

        for record in records:
            relative_path = _safe_relative_path(record.file_name)
            image_path = dataset_dir.joinpath(*relative_path.parts)

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image referenced by {annotation_path} is missing: {image_path}"
                )


def _cache_is_ready(dataset_dir: Path, expected_uri: str) -> bool:
    """Return whether a previous download is complete and valid."""
    marker = dataset_dir / CACHE_MARKER

    if not marker.is_file():
        return False

    if marker.read_text(encoding="utf-8").strip() != expected_uri:
        return False

    try:
        _validate_dataset_content(dataset_dir)
    except (OSError, ValueError, TypeError):
        LOGGER.warning("Cached dataset is incomplete or invalid: %s", dataset_dir)
        return False

    return True


def _list_s3_objects(client: BaseClient, prefix: str) -> list[tuple[str, int]]:
    """List all object keys and their sizes under an S3 prefix."""
    objects: list[tuple[str, int]] = []
    continuation_token: str | None = None

    while True:
        request: dict[str, str] = {
            "Bucket": S3_BUCKET,
            "Prefix": f"{prefix}/",
        }

        if continuation_token is not None:
            request["ContinuationToken"] = continuation_token

        response = client.list_objects_v2(**request)
        contents = response.get("Contents", [])

        if isinstance(contents, list):
            for item in contents:
                if not isinstance(item, dict):
                    continue

                key = item.get("Key")
                size = item.get("Size")

                if isinstance(key, str) and isinstance(size, int):
                    objects.append((key, size))

        if not response.get("IsTruncated", False):
            break

        next_token = response.get("NextContinuationToken")
        if not isinstance(next_token, str):
            raise TypeError("S3 listing is truncated but has no continuation token")

        continuation_token = next_token

    return objects


def _download_object(
    client: BaseClient,
    key: str,
    destination: Path,
) -> None:
    """Download one S3 object to its local destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(S3_BUCKET, key, str(destination))


def download_dataset(
    s3_path: str,
    cache_root: Path | None = None,
) -> Path:
    """Download a versioned dataset from S3 unless a complete local copy exists.

    The returned directory contains annotations/, images/, stats.json,
    and manifest.json. The downloaded files preserve their S3-relative paths.
    """
    prefix = _normalize_s3_prefix(s3_path)
    base_dir = cache_root if cache_root is not None else CACHE_ROOT
    dataset_dir = base_dir.joinpath(*PurePosixPath(prefix).parts)
    expected_uri = f"s3://{S3_BUCKET}/{prefix}"

    if _cache_is_ready(dataset_dir, expected_uri):
        LOGGER.info("Using cached dataset at %s", dataset_dir)
        return dataset_dir

    # Do not trust an incomplete cache or a cache from another bucket/version.
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    dataset_dir.mkdir(parents=True, exist_ok=True)

    client = create_s3_client()
    objects = _list_s3_objects(client, prefix)

    if not objects:
        raise FileNotFoundError(f"No objects found at {expected_uri}/")

    prefix_with_separator = f"{prefix}/"
    downloads: list[tuple[str, Path]] = []

    for key, expected_size in objects:
        if not key.startswith(prefix_with_separator):
            raise ValueError(f"S3 returned an object outside the requested prefix: {key}")

        relative_key = key[len(prefix_with_separator) :]
        relative_path = _safe_relative_path(relative_key)
        destination = dataset_dir.joinpath(*relative_path.parts)

        if destination.is_file() and destination.stat().st_size == expected_size:
            continue

        downloads.append((key, destination))

    LOGGER.info(
        "Downloading %d of %d objects from %s",
        len(downloads),
        len(objects),
        expected_uri,
    )

    if downloads:
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
            futures = [
                executor.submit(_download_object, client, key, destination)
                for key, destination in downloads
            ]

            for count, future in enumerate(as_completed(futures), start=1):
                future.result()

                if count % 1000 == 0 or count == len(futures):
                    LOGGER.info("Downloaded %d/%d objects", count, len(futures))

    # Only mark the cache complete after all metadata and image references pass.
    _validate_dataset_content(dataset_dir)
    (dataset_dir / CACHE_MARKER).write_text(expected_uri + "\n", encoding="utf-8")

    LOGGER.info("Dataset is ready at %s", dataset_dir)
    return dataset_dir


class CIFAR10Dataset(Dataset[tuple[Tensor, int]]):
    """PyTorch dataset backed by the versioned PNG files and JSON annotations."""

    def __init__(
        self,
        dataset_dir: Path,
        split: str,
        mean: list[float],
        std: list[float],
        max_samples: int | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unsupported dataset split: {split}")

        if max_samples is not None and max_samples < 1:
            raise ValueError("max_samples must be positive or None")

        annotation_path = dataset_dir / "annotations" / f"{split}.json"
        records = _load_records(annotation_path)

        if max_samples is not None:
            records = records[:max_samples]

        self.dataset_dir = dataset_dir
        self.records = records
        self.transform = Compose(
            [
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        record = self.records[index]
        relative_path = _safe_relative_path(record.file_name)
        image_path = self.dataset_dir.joinpath(*relative_path.parts)

        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))

        if not isinstance(tensor, Tensor):
            raise TypeError(f"Image transformation did not return a tensor: {image_path}")

        return tensor, record.label


def create_datasets(
    s3_path: str,
    max_train_samples: int | None = None,
) -> tuple[CIFAR10Dataset, CIFAR10Dataset, CIFAR10Dataset]:
    """Download a dataset version and create train, validation, and test datasets."""
    dataset_dir = download_dataset(s3_path)
    mean, std = load_normalization_stats(dataset_dir / "stats.json")

    train_dataset = CIFAR10Dataset(
        dataset_dir,
        "train",
        mean,
        std,
        max_samples=max_train_samples,
    )
    val_dataset = CIFAR10Dataset(dataset_dir, "val", mean, std)
    test_dataset = CIFAR10Dataset(dataset_dir, "test", mean, std)

    return train_dataset, val_dataset, test_dataset
