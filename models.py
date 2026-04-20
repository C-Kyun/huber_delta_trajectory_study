"""Model definitions for trajectory reconstruction experiments."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    """Simple MLP regressor with Tanh activations for 3D trajectory prediction."""

    def __init__(self, input_dim: int, hidden_width: int, depth: int, output_dim: int = 3) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_width))
            layers.append(nn.Tanh())
            in_dim = hidden_width
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Use a single initialization strategy across all experiments."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


def build_mlp(hidden_width: int, depth: int) -> MLPRegressor:
    """Factory for default trajectory model architecture."""
    return MLPRegressor(input_dim=1, hidden_width=hidden_width, depth=depth, output_dim=3)
