"""Generalized matrix factorization for explicit-feedback ratings."""

import torch
from torch import nn


class GeneralizedMatrixFactorization(nn.Module):
    """Learn a weighted interaction over element-wise embeddings."""

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        embedding_dim: int = 32,
        global_mean: float = 0.0,
    ) -> None:
        super().__init__()
        if min(num_users, num_movies, embedding_dim) <= 0:
            raise ValueError("model dimensions must be positive")

        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_dim = embedding_dim
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.output = nn.Linear(embedding_dim, 1)
        self.register_buffer(
            "global_mean",
            torch.tensor(float(global_mean), dtype=torch.float32),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize embeddings and the output layer."""
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.movie_embedding.weight, std=0.05)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        user_indices: torch.Tensor,
        movie_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return one unconstrained predicted rating per pair."""
        if user_indices.shape != movie_indices.shape:
            raise ValueError("user_indices and movie_indices must align")
        interaction = self.user_embedding(
            user_indices
        ) * self.movie_embedding(movie_indices)
        return self.global_mean + self.output(interaction).squeeze(-1)

    def config(self) -> dict[str, int | float | str]:
        """Return enough metadata to reconstruct the model."""
        return {
            "model": "gmf",
            "num_users": self.num_users,
            "num_movies": self.num_movies,
            "embedding_dim": self.embedding_dim,
            "global_mean": float(self.global_mean.item()),
        }
