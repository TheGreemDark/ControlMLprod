"""Save experiment artifacts locally and upload them to S3."""

import json
from pathlib import Path

import torch
import yaml
from botocore.client import BaseClient
from torch import nn

from controlmlprod.data.config import DATA_DIR, S3_BUCKET
from controlmlprod.data.s3_client import create_s3_client
from controlmlprod.experiments.config import ExperimentConfig


def upload_artifacts(
    model: nn.Module,
    config: ExperimentConfig,
    metrics: dict[str, object],
) -> str:
    """Save model, config and metrics, then publish them to S3."""
    output_dir = DATA_DIR / "experiment_artifacts" / config.experiment_name / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.pt"
    config_path = output_dir / "config.yaml"
    metrics_path = output_dir / "metrics.json"

    torch.save(model.state_dict(), model_path)
    config_path.write_text(
        yaml.safe_dump(
            config.to_dict(),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    prefix = f"models/{config.experiment_name}/{config.run_name}"
    client: BaseClient = create_s3_client()

    for path in (model_path, config_path, metrics_path):
        client.upload_file(
            str(path),
            S3_BUCKET,
            f"{prefix}/{path.name}",
        )

    return f"s3://{S3_BUCKET}/{prefix}/"


def artifact_directory(config: ExperimentConfig) -> Path:
    """Return the local directory used for this run's intermediate checkpoint."""
    return DATA_DIR / "experiment_checkpoints" / config.experiment_name / config.run_name
