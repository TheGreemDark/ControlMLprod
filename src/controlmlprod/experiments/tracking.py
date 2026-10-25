"""Weights & Biases integration for experiment tracking."""

import os
from typing import Literal, cast

from wandb.sdk.wandb_run import Run

import wandb
from controlmlprod.experiments.config import ExperimentConfig

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


def start_wandb_run(config: ExperimentConfig, git_commit: str) -> Run:
    """Start a tracked run and record its complete configuration."""
    project = os.getenv("WANDB_PROJECT", "controlmlprod-hw3")
    entity = os.getenv("WANDB_ENTITY", "").strip() or None
    mode_value = os.getenv("WANDB_MODE", "online").strip()
    allowed_modes = {"online", "offline", "disabled", "shared"}

    if mode_value not in allowed_modes:
        raise ValueError(
            f"Unsupported WANDB_MODE: {mode_value!r}. "
            f"Expected one of: {', '.join(sorted(allowed_modes))}"
        )

    mode = cast(
        Literal["online", "offline", "disabled", "shared"],
        mode_value,
    )

    if mode == "online" and not os.getenv("WANDB_API_KEY"):
        raise RuntimeError(
            "WANDB_API_KEY is missing. Add it to the local .env file "
            "or explicitly select WANDB_MODE=offline for offline testing."
        )

    tracked_config = config.to_dict()
    tracked_config["git_commit"] = git_commit
    tracked_config["dataset_s3_path"] = config.data.s3_path

    run = wandb.init(
        project=project,
        entity=entity,
        name=config.run_name,
        group=config.experiment_name,
        config=tracked_config,
        mode=mode,
    )

    if run is None:
        raise RuntimeError("Weights & Biases did not create a run.")

    return run


def log_validation_confusion_matrix(
    run: Run,
    targets: list[int],
    predictions: list[int],
) -> None:
    """Log the best checkpoint's validation confusion matrix to W&B."""
    plot = wandb.plot.confusion_matrix(
        y_true=targets,
        preds=predictions,
        class_names=CLASS_NAMES,
        title="Validation confusion matrix — best checkpoint",
    )
    run.log({"val/confusion_matrix": plot})
