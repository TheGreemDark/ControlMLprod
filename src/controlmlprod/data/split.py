import json
import logging
import os
import shutil
from pathlib import Path
from typing import cast

from sklearn.model_selection import train_test_split

from controlmlprod.data.config import DATA_DIR

LOGGER = logging.getLogger(__name__)

OUTPUT_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = OUTPUT_DIR / "annotations"

SOURCE_FILE = ANNOTATIONS_DIR / "train_all.json"
TRAIN_FILE = ANNOTATIONS_DIR / "train.json"
VAL_FILE = ANNOTATIONS_DIR / "val.json"

TEST_SIZE = int(os.getenv("VAL_SIZE", "5000"))
SPLIT_SEED = int(os.getenv("SPLIT_SEED", "42"))


def load_records(path: Path) -> list[dict[str, object]]:
    """Load and validate annotation records from a JSON file."""
    with path.open(encoding="utf-8") as file:
        value = json.load(file)

    if not isinstance(value, list):
        raise TypeError(f"Expected a JSON list in {path}")

    records: list[dict[str, object]] = []

    for item in value:
        if not isinstance(item, dict):
            raise TypeError(f"Invalid annotation record in {path}")

        records.append(cast(dict[str, object], item))

    return records


def save_records(path: Path, records: list[dict[str, object]]) -> None:
    """Save annotation records to a JSON file."""
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def split_dataset() -> None:
    """Create a deterministic stratified train/validation split."""
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Preserve the original 50,000 training records before splitting.
    if not SOURCE_FILE.exists():
        if not TRAIN_FILE.exists():
            raise FileNotFoundError(f"Missing {TRAIN_FILE}. Run the conversion step first.")

        original_records = load_records(TRAIN_FILE)

        if len(original_records) != 50_000:
            raise ValueError(
                "Cannot create train_all.json: train.json does not contain "
                "the original 50,000 records. Run conversion again."
            )

        shutil.copy2(TRAIN_FILE, SOURCE_FILE)

    records = load_records(SOURCE_FILE)

    if len(records) != 50_000:
        raise ValueError(f"Expected 50,000 original training records, found {len(records)}")

    if not 0 < TEST_SIZE < len(records):
        raise ValueError("VAL_SIZE must be between 1 and 49,999")

    labels: list[int] = []

    for record in records:
        label = record.get("label")

        if not isinstance(label, int) or isinstance(label, bool):
            raise TypeError(f"Invalid label in training record: {label!r}")

        labels.append(label)

    train_records, val_records = train_test_split(
        records,
        test_size=TEST_SIZE,
        random_state=SPLIT_SEED,
        stratify=labels,
    )

    save_records(TRAIN_FILE, train_records)
    save_records(VAL_FILE, val_records)

    LOGGER.info(
        "Split complete: train=%d, val=%d, seed=%d",
        len(train_records),
        len(val_records),
        SPLIT_SEED,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    split_dataset()


if __name__ == "__main__":
    main()
