"""Evaluation metrics for image classification experiments."""

from dataclasses import dataclass

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    accuracy: float
    macro_f1: float
    targets: list[int]
    predictions: list[int]
    confusion: list[list[int]]


def evaluate_model(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    num_classes: int = 10,
) -> EvaluationResult:
    """Evaluate a model without updating its parameters."""
    model.eval()

    loss_sum = 0.0
    sample_count = 0
    targets: list[int] = []
    predictions: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = int(labels.size(0))
            loss_sum += float(loss.item()) * batch_size
            sample_count += batch_size

            targets.extend(int(value) for value in labels.tolist())
            predictions.extend(int(value) for value in logits.argmax(dim=1).tolist())

    if sample_count == 0:
        raise ValueError("Cannot evaluate an empty dataset")

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=list(range(num_classes)),
    )

    return EvaluationResult(
        loss=loss_sum / sample_count,
        accuracy=float(accuracy_score(targets, predictions)),
        macro_f1=float(
            f1_score(
                targets,
                predictions,
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            )
        ),
        targets=targets,
        predictions=predictions,
        confusion=[[int(value) for value in row] for row in matrix.tolist()],
    )
