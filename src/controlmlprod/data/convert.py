import hashlib
import json
import logging
import pickle
import tarfile
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from controlmlprod.data.config import DATA_DIR
from controlmlprod.data.download_raw import download_raw_archive

LOGGER = logging.getLogger(__name__)

OUTPUT_DIR = DATA_DIR / "processed"
IMAGES_DIR = OUTPUT_DIR / "images"
ANNOTATIONS_DIR = OUTPUT_DIR / "annotations"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def read_pickle_member(archive: tarfile.TarFile, name: str) -> dict[str, object]:
    member = archive.getmember(f"cifar-10-batches-py/{name}")
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"Cannot read archive member: {name}")

    with stream:
        value = pickle.load(stream, encoding="bytes")

    if not isinstance(value, dict):
        raise TypeError(f"Unexpected CIFAR-10 data format in {name}")

    result: dict[str, object] = {}
    for key, item in value.items():
        normalized_key = key.decode("ascii") if isinstance(key, bytes) else key
        if isinstance(normalized_key, str):
            result[normalized_key] = item

    return result


def convert_batch(
    data: NDArray[np.uint8],
    labels: list[int],
    split: str,
    start_index: int,
) -> list[dict[str, object]]:
    if data.ndim != 2 or data.shape[1] != 3072:
        raise ValueError(f"Unexpected image data shape: {data.shape}")

    if len(data) != len(labels):
        raise ValueError("The number of images and labels does not match")

    split_dir = IMAGES_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []

    for offset, (pixels, label) in enumerate(zip(data, labels, strict=True)):
        if not 0 <= label < len(CLASS_NAMES):
            raise ValueError(f"Invalid CIFAR-10 label: {label}")

        image_id = f"{split}_{start_index + offset:06d}"
        file_name = f"images/{split}/{image_id}.png"
        image_path = OUTPUT_DIR / file_name

        image_array = pixels.reshape(3, 32, 32).transpose(1, 2, 0)
        image = Image.fromarray(cast(NDArray[np.uint8], image_array), mode="RGB")
        image.save(image_path, format="PNG")

        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()

        records.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "label": label,
                "label_name": CLASS_NAMES[label],
                "sha256": digest,
            }
        )

    return records


def write_annotations(name: str, records: list[dict[str, object]]) -> None:
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = ANNOTATIONS_DIR / name
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Wrote %s records to %s", len(records), path)


def convert_dataset(archive_path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_records: list[dict[str, object]] = []
    test_records: list[dict[str, object]] = []

    with tarfile.open(archive_path, mode="r:gz") as archive:
        for batch_number in range(1, 6):
            name = f"data_batch_{batch_number}"
            batch = read_pickle_member(archive, name)

            data = np.asarray(batch["data"], dtype=np.uint8)
            raw_labels = batch.get("labels")
            if not isinstance(raw_labels, list):
                raise TypeError(f"Missing labels in {name}")

            labels = [int(label) for label in raw_labels]
            train_records.extend(
                convert_batch(
                    data,
                    labels,
                    split="train",
                    start_index=len(train_records),
                )
            )
            LOGGER.info("Converted %s", name)

        test_batch = read_pickle_member(archive, "test_batch")
        test_data = np.asarray(test_batch["data"], dtype=np.uint8)
        raw_test_labels = test_batch.get("labels")
        if not isinstance(raw_test_labels, list):
            raise TypeError("Missing labels in test_batch")

        test_labels = [int(label) for label in raw_test_labels]
        test_records = convert_batch(
            test_data,
            test_labels,
            split="test",
            start_index=0,
        )

    if len(train_records) != 50_000 or len(test_records) != 10_000:
        raise ValueError(
            f"Unexpected dataset size: train={len(train_records)}, test={len(test_records)}"
        )

    write_annotations("train.json", train_records)
    write_annotations("test.json", test_records)

    LOGGER.info(
        "Conversion complete: train=%s, test=%s",
        len(train_records),
        len(test_records),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    archive_path = download_raw_archive()
    convert_dataset(archive_path)


if __name__ == "__main__":
    main()
