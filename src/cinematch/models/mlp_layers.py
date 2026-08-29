"""Reusable multilayer perceptron block for NCF models."""

from collections.abc import Sequence

import torch
from torch import nn


class MLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        layers: Sequence[int],
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or not layers or any(size <= 0 for size in layers):
            raise ValueError("input_dim and all hidden layer sizes must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        modules: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in layers:
            modules.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    # LayerNorm also works when the final training batch has one row.
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                ]
            )
            current_dim = hidden_dim

        self.network = nn.Sequential(*modules)
        self.output_dim = layers[-1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)
