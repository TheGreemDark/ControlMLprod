import json
import logging

import numpy as np
from PIL import Image

from controlmlprod.data.config import DATA_DIR

LOGGER = logging.getLogger(__name__)

OUTPUT_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = OUTPUT_DIR / "annotations"
TRAIN_FILE = ANNOTATIONS_DIR / "train.json"
STATS_FILE = OUTPUT_DIR / "stats.json"


def calculate_statistics() -> dict[str, object]:
    with TRAIN_FILE.open(encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list) or not records:
        raise ValueError("train.json must contain a non-empty JSON list")

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_squared_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("file_name"), str):
            raise TypeError("Invalid record in train.json")

        image_path = OUTPUT_DIR / record["file_name"]

        with Image.open(image_path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0

        if pixels.shape != (32, 32, 3):
            raise ValueError(f"Unexpected image shape: {pixels.shape}")

        channel_sum += pixels.sum(axis=(0, 1))
        channel_squared_sum += np.square(pixels).sum(axis=(0, 1))
        pixel_count += pixels.shape[0] * pixels.shape[1]

    mean = channel_sum / pixel_count
    variance = np.maximum(channel_squared_sum / pixel_count - np.square(mean), 0)
    std = np.sqrt(variance)

    stats: dict[str, object] = {
        "dataset": "CIFAR-10",
        "split": "train",
        "num_images": len(records),
        "pixel_range": [0.0, 1.0],
        "std_method": "population",
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_FILE.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    LOGGER.info(
        "Saved RGB statistics to %s: mean=%s, std=%s",
        STATS_FILE,
        stats["mean"],
        stats["std"],
    )
    return stats


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    calculate_statistics()


if __name__ == "__main__":
    main()
