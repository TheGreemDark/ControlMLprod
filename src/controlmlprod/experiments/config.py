"""Typed loading and validation of single-run experiment YAML files."""

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import yaml


@dataclass(frozen=True)
class DataConfig:
    s3_path: str
    max_train_samples: int | None


@dataclass(frozen=True)
class ModelConfig:
    type: str
    params: dict[str, object]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    num_workers: int


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    run_name: str
    seed: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig

    def to_dict(self) -> dict[str, object]:
        """Return a serializable dictionary for tracking and artifacts."""
        return cast(dict[str, object], asdict(self))


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a YAML mapping")

    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field} must contain string keys")

    return cast(dict[str, object], value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _positive_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a positive number")

    result = float(value)
    if result <= 0:
        raise ValueError(f"{field} must be a positive number")
    return result


def load_config(path: Path) -> ExperimentConfig:
    """Read and validate a single-experiment YAML configuration."""
    with path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    root = _mapping(raw, "config")

    experiment_name = _string(root.get("experiment_name"), "experiment_name")
    run_name = _string(root.get("run_name"), "run_name")

    # These values become part of artifact paths in S3.
    safe_name = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    if not safe_name.fullmatch(experiment_name):
        raise ValueError("experiment_name contains unsupported characters")
    if not safe_name.fullmatch(run_name):
        raise ValueError("run_name contains unsupported characters")

    seed = _integer(root.get("seed"), "seed", 0)

    raw_data = _mapping(root.get("data"), "data")
    s3_path = _string(raw_data.get("s3_path"), "data.s3_path")
    max_samples_raw = raw_data.get("max_train_samples")

    max_train_samples: int | None
    if max_samples_raw is None:
        max_train_samples = None
    else:
        max_train_samples = _integer(
            max_samples_raw,
            "data.max_train_samples",
            1,
        )

    raw_model = _mapping(root.get("model"), "model")
    model_type = _string(raw_model.get("type"), "model.type")
    if model_type not in {"linear", "simple_cnn"}:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_params = _mapping(raw_model.get("params", {}), "model.params")

    raw_train = _mapping(root.get("train"), "train")
    train = TrainConfig(
        epochs=_integer(raw_train.get("epochs"), "train.epochs", 1),
        batch_size=_integer(
            raw_train.get("batch_size"),
            "train.batch_size",
            1,
        ),
        lr=_positive_float(raw_train.get("lr"), "train.lr"),
        num_workers=_integer(
            raw_train.get("num_workers", 0),
            "train.num_workers",
            0,
        ),
    )

    return ExperimentConfig(
        experiment_name=experiment_name,
        run_name=run_name,
        seed=seed,
        data=DataConfig(
            s3_path=s3_path,
            max_train_samples=max_train_samples,
        ),
        model=ModelConfig(type=model_type, params=model_params),
        train=train,
    )
