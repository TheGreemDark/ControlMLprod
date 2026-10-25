"""PyTorch model architectures for CIFAR-10 experiments."""

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn


class LinearBaseline(nn.Module):
    """A single linear layer applied to flattened image pixels."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.classifier = nn.Linear(3 * 32 * 32, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        return cast(Tensor, self.classifier(torch.flatten(images, start_dim=1)))


class SimpleCNN(nn.Module):
    """A compact CNN with two convolutional layers."""

    def __init__(
        self,
        channels: Sequence[int] = (32, 64),
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        if len(channels) != 2:
            raise ValueError("SimpleCNN requires exactly two channel sizes")

        first_channels, second_channels = channels

        if first_channels <= 0 or second_channels <= 0:
            raise ValueError("Channel sizes must be positive")

        self.features = nn.Sequential(
            nn.Conv2d(3, first_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                first_channels,
                second_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Linear(second_channels * 8 * 8, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        features = self.features(images)
        return cast(Tensor, self.classifier(torch.flatten(features, start_dim=1)))


def build_model(
    model_type: str,
    params: dict[str, object],
    num_classes: int = 10,
) -> nn.Module:
    """Build a supported model from configuration parameters."""
    if model_type == "linear":
        return LinearBaseline(num_classes=num_classes)

    if model_type == "simple_cnn":
        raw_channels = params.get("channels", [32, 64])

        if not isinstance(raw_channels, list):
            raise TypeError("model.params.channels must be a list of integers")

        channels: list[int] = []
        for value in raw_channels:
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError("model.params.channels must contain integers")
            channels.append(value)

        return SimpleCNN(channels=channels, num_classes=num_classes)

    raise ValueError(f"Unsupported model type: {model_type}")
