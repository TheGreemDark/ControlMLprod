"""Run one reproducible CIFAR-10 training experiment."""

import argparse
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from controlmlprod.data.config import PROJECT_ROOT, S3_BUCKET
from controlmlprod.experiments.artifacts import (
    artifact_directory,
    upload_artifacts,
)
from controlmlprod.experiments.config import ExperimentConfig, load_config
from controlmlprod.experiments.dataset import create_datasets
from controlmlprod.experiments.metrics import EvaluationResult, evaluate_model
from controlmlprod.experiments.models import build_model
from controlmlprod.experiments.tracking import (
    log_validation_confusion_matrix,
    start_wandb_run,
)


def get_git_commit() -> str:
    """Return the commit used to launch the experiment."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"

    return result.stdout.strip()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible CPU experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, int]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Perform one training epoch and return mean training loss."""
    model.train()
    total_loss = 0.0
    sample_count = 0

    for images, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = int(labels.size(0))
        total_loss += float(loss.item()) * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("Cannot train on an empty dataset")

    return total_loss / sample_count


def metric_values(result: EvaluationResult) -> dict[str, float]:
    """Extract scalar evaluation metrics."""
    return {
        "loss": result.loss,
        "accuracy": result.accuracy,
        "macro_f1": result.macro_f1,
    }


def run_training(config: ExperimentConfig) -> None:
    """Train, select the best validation checkpoint, and evaluate on test."""
    set_seed(config.seed)
    num_threads = max(1, int(os.environ.get("TORCH_NUM_THREADS", "2")))
    torch.set_num_threads(num_threads)

    git_commit = get_git_commit()
    run = start_wandb_run(config, git_commit)

    try:
        train_dataset, val_dataset, test_dataset = create_datasets(
            s3_path=config.data.s3_path,
            max_train_samples=config.data.max_train_samples,
        )

        generator = torch.Generator()
        generator.manual_seed(config.seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=config.train.num_workers,
            generator=generator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.num_workers,
        )

        model = build_model(
            model_type=config.model.type,
            params=config.model.params,
            num_classes=10,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.train.lr,
        )

        checkpoint_dir = artifact_directory(config)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "best-checkpoint.pt"

        best_val_loss = float("inf")
        best_epoch = 0
        best_validation: EvaluationResult | None = None
        training_started = time.perf_counter()

        for epoch in range(1, config.train.epochs + 1):
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
            )
            validation = evaluate_model(model, val_loader, criterion)

            run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": validation.loss,
                    "val/accuracy": validation.accuracy,
                    "val/macro_f1": validation.macro_f1,
                }
            )

            print(
                f"Epoch {epoch}/{config.train.epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={validation.loss:.4f}, "
                f"val_accuracy={validation.accuracy:.4f}, "
                f"val_macro_f1={validation.macro_f1:.4f}"
            )

            if validation.loss < best_val_loss:
                best_val_loss = validation.loss
                best_epoch = epoch
                best_validation = validation
                torch.save(model.state_dict(), checkpoint_path)

        training_time = time.perf_counter() - training_started

        if best_validation is None:
            raise RuntimeError("Training did not produce a validation checkpoint")

        # Test is evaluated only after selecting the best checkpoint by val loss.
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        test_result = evaluate_model(model, test_loader, criterion)

        run.log(
            {
                "test/loss": test_result.loss,
                "test/accuracy": test_result.accuracy,
                "test/macro_f1": test_result.macro_f1,
            }
        )
        log_validation_confusion_matrix(
            run,
            best_validation.targets,
            best_validation.predictions,
        )

        dataset_uri = f"s3://{S3_BUCKET}/{config.data.s3_path.strip('/')}"
        metrics: dict[str, object] = {
            "experiment_name": config.experiment_name,
            "run_name": config.run_name,
            "seed": config.seed,
            "git_commit": git_commit,
            "dataset_s3_path": dataset_uri,
            "best_checkpoint_epoch": best_epoch,
            "training_time_seconds": training_time,
            "val": metric_values(best_validation),
            "test": metric_values(test_result),
        }

        artifact_uri = upload_artifacts(model, config, metrics)

        run.summary["best_checkpoint_epoch"] = best_epoch
        run.summary["training_time_seconds"] = training_time
        run.summary["val/loss"] = best_validation.loss
        run.summary["val/accuracy"] = best_validation.accuracy
        run.summary["val/macro_f1"] = best_validation.macro_f1
        run.summary["test/loss"] = test_result.loss
        run.summary["test/accuracy"] = test_result.accuracy
        run.summary["test/macro_f1"] = test_result.macro_f1
        run.summary["model_artifact_uri"] = artifact_uri

        print(f"Best checkpoint epoch: {best_epoch}")
        print(f"Training time: {training_time:.2f}s")
        print(f"Model artifacts: {artifact_uri}")

    finally:
        run.finish()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration for one experiment",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
