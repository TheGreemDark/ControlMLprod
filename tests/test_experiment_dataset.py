"""Tests for the experiment dataset loader."""

import json
from pathlib import Path

import pytest
from PIL import Image

from controlmlprod.experiments.dataset import (
    CIFAR10Dataset,
    _normalize_s3_prefix,
    _safe_relative_path,
    load_normalization_stats,
)


def test_normalize_s3_prefix_removes_trailing_slash() -> None:
    assert _normalize_s3_prefix("processed/cifar10/v1/") == "processed/cifar10/v1"


@pytest.mark.parametrize(
    "value",
    ["", "/absolute/path", "../outside", "processed/../outside", r"images\train.png"],
)
def test_normalize_s3_prefix_rejects_unsafe_paths(value: str) -> None:
    with pytest.raises(ValueError):
        _normalize_s3_prefix(value)


@pytest.mark.parametrize(
    "value",
    ["", "/etc/passwd", "../secret.png", "images/../secret.png", r"images\train.png"],
)
def test_safe_relative_path_rejects_unsafe_paths(value: str) -> None:
    with pytest.raises(ValueError):
        _safe_relative_path(value)


def test_load_normalization_stats_reads_rgb_triplets(tmp_path: Path) -> None:
    path = tmp_path / "stats.json"
    path.write_text(
        json.dumps({"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6]}),
        encoding="utf-8",
    )

    mean, std = load_normalization_stats(path)

    assert mean == [0.1, 0.2, 0.3]
    assert std == [0.4, 0.5, 0.6]


def test_load_normalization_stats_rejects_zero_standard_deviation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stats.json"
    path.write_text(
        json.dumps({"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.0, 0.6]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="positive"):
        load_normalization_stats(path)


def test_cifar10_dataset_loads_and_normalizes_an_image(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    annotations_dir = dataset_dir / "annotations"
    image_path = dataset_dir / "images" / "train" / "sample.png"
    annotations_dir.mkdir(parents=True)
    image_path.parent.mkdir(parents=True)

    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(image_path)

    annotations = [
        {
            "image_id": "sample",
            "file_name": "images/train/sample.png",
            "label": 2,
            "label_name": "bird",
            "sha256": "unused-in-this-test",
        }
    ]
    (annotations_dir / "train.json").write_text(
        json.dumps(annotations),
        encoding="utf-8",
    )

    dataset = CIFAR10Dataset(
        dataset_dir=dataset_dir,
        split="train",
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )

    image, label = dataset[0]

    assert len(dataset) == 1
    assert image.shape == (3, 32, 32)
    assert image[:, 0, 0].tolist() == pytest.approx([1.0, -1.0, -1.0])
    assert label == 2


def test_cifar10_dataset_rejects_unknown_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported dataset split"):
        CIFAR10Dataset(
            dataset_dir=tmp_path,
            split="unknown",
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )
