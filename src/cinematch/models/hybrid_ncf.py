"""Hybrid NCF model that combines ID embeddings with side features."""

from collections.abc import Sequence

import torch
from torch import nn

from cinematch.models.embeddings import EmbeddingLayer
from cinematch.models.mlp_layers import MLPBlock


class HybridNCF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        side_feature_dim: int,
        embedding_dim: int = 32,
        layers: Sequence[int] = (64, 32, 16),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if side_feature_dim <= 0:
            raise ValueError("side_feature_dim must be positive")

        self.side_feature_dim = side_feature_dim
        self.user_embed = EmbeddingLayer(num_users, embedding_dim)
        self.item_embed = EmbeddingLayer(num_items, embedding_dim)

        total_input_dim = (embedding_dim * 2) + side_feature_dim
        self.mlp = MLPBlock(
            input_dim=total_input_dim,
            layers=layers,
            dropout=dropout,
        )
        self.prediction_head = nn.Linear(self.mlp.output_dim, 1)

    def forward(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
        side_features: torch.Tensor,
    ) -> torch.Tensor:
        if side_features.ndim != 2:
            raise ValueError("side_features must have shape [batch, features]")
        if side_features.shape[0] != user_indices.shape[0]:
            raise ValueError("side_features batch size must match user indices")
        if side_features.shape[1] != self.side_feature_dim:
            raise ValueError(
                f"expected {self.side_feature_dim} side features, "
                f"received {side_features.shape[1]}"
            )

        u_vector = self.user_embed(user_indices)
        i_vector = self.item_embed(item_indices)

        x = torch.cat([u_vector, i_vector, side_features], dim=-1)
        x = self.mlp(x)
        raw_score = self.prediction_head(x)
        rating = 1.0 + 4.0 * torch.sigmoid(raw_score)
        return rating.squeeze(-1)
