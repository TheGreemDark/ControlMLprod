"""Tests for hyperparameter grid generation."""

from pathlib import Path

import yaml

from controlmlprod.experiments.run_grid import build_grid_configs


def test_grid_generates_all_unique_combinations(tmp_path: Path) -> None:
    base = {
        "experiment_name": "cifar10-simple-cnn",
        "run_name": "base-run",
        "seed": 42,
        "data": {
            "s3_path": "processed/cifar10/v1",
            "max_train_samples": 2048,
        },
        "model": {
            "type": "simple_cnn",
            "params": {"channels": [32, 64]},
        },
        "train": {
            "epochs": 3,
            "batch_size": 128,
            "lr": 0.001,
            "num_workers": 0,
        },
    }
    grid = {
        "experiment_name": "cifar10-cnn-grid",
        "base": "base.yaml",
        "grid": {
            "train.lr": [0.0003, 0.001, 0.003],
            "train.batch_size": [64, 128],
        },
    }

    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump(base),
        encoding="utf-8",
    )
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(yaml.safe_dump(grid), encoding="utf-8")

    configs = build_grid_configs(grid_path)

    assert len(configs) == 6
    names = [config["run_name"] for config in configs]
    assert len(set(names)) == 6

    for config in configs:
        assert config["experiment_name"] == "cifar10-cnn-grid"
        assert str(config["run_name"]).startswith("cnn-")
        assert config["seed"] == 42

    combinations = {
        (
            config["train"]["lr"],  # type: ignore[index]
            config["train"]["batch_size"],  # type: ignore[index]
        )
        for config in configs
    }
    assert combinations == {
        (0.0003, 64),
        (0.0003, 128),
        (0.001, 64),
        (0.001, 128),
        (0.003, 64),
        (0.003, 128),
    }
