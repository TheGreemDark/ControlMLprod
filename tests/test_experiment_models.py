"""Tests for experiment model architectures."""

import pytest
import torch

from controlmlprod.experiments.models import build_model


@pytest.mark.parametrize(
    ("model_type", "params"),
    [
        ("linear", {}),
        ("simple_cnn", {"channels": [8, 16]}),
    ],
)
def test_model_returns_ten_class_logits(
    model_type: str,
    params: dict[str, object],
) -> None:
    model = build_model(model_type, params)
    images = torch.randn(2, 3, 32, 32)

    logits = model(images)

    assert logits.shape == (2, 10)


def test_unknown_model_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported model type"):
        build_model("unknown", {})


def test_cnn_rejects_invalid_channel_configuration() -> None:
    with pytest.raises(TypeError, match="list of integers"):
        build_model("simple_cnn", {"channels": "32,64"})
