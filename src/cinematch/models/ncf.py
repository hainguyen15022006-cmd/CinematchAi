"""Neural Collaborative Filtering model for explicit MovieLens ratings."""

from collections.abc import Sequence

import torch
from torch import nn

from cinematch.models.embeddings import EmbeddingLayer
from cinematch.models.mlp_layers import MLPBlock


class NCF(nn.Module):
    """Predict a rating from learned user and movie embeddings."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        layers: Sequence[int] = (64, 32, 16),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_movies = num_items
        self.embedding_dim = embedding_dim
        self.layers = tuple(layers)
        self.dropout = float(dropout)
        self.user_embed = EmbeddingLayer(num_users, embedding_dim)
        self.item_embed = EmbeddingLayer(num_items, embedding_dim)

        self.mlp = MLPBlock(
            input_dim=embedding_dim * 2,
            layers=layers,
            dropout=dropout,
        )
        self.prediction_head = nn.Linear(self.mlp.output_dim, 1)

    def forward(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
    ) -> torch.Tensor:
        u_vector = self.user_embed(user_indices)
        i_vector = self.item_embed(item_indices)

        x = torch.cat([u_vector, i_vector], dim=-1)
        x = self.mlp(x)
        raw_score = self.prediction_head(x)
        # Sigmoid maps an unbounded logit to the explicit rating range [1, 5].
        rating = 1.0 + 4.0 * torch.sigmoid(raw_score)
        return rating.squeeze(-1)

    def config(self) -> dict[str, int | float | str | list[int]]:
        """Return a checkpoint configuration compatible with Predictor."""
        return {
            "model": "ncf",
            "num_users": self.num_users,
            "num_movies": self.num_movies,
            "embedding_dim": self.embedding_dim,
            "layers": list(self.layers),
            "dropout": self.dropout,
        }
