"""Biased matrix factorization for explicit-feedback ratings."""

import torch
from torch import nn


class MatrixFactorization(nn.Module):
    """Predict ratings from a latent dot product and learned biases."""

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
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)
        self.register_buffer(
            "global_mean",
            torch.tensor(float(global_mean), dtype=torch.float32),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize latent vectors and biases."""
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.movie_embedding.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(
        self,
        user_indices: torch.Tensor,
        movie_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return one unconstrained predicted rating per pair."""
        if user_indices.shape != movie_indices.shape:
            raise ValueError("user_indices and movie_indices must align")
        user_vectors = self.user_embedding(user_indices)
        movie_vectors = self.movie_embedding(movie_indices)
        interaction = (user_vectors * movie_vectors).sum(dim=-1)
        user_bias = self.user_bias(user_indices).squeeze(-1)
        movie_bias = self.movie_bias(movie_indices).squeeze(-1)
        return self.global_mean + user_bias + movie_bias + interaction

    def config(self) -> dict[str, int | float | str]:
        """Return enough metadata to reconstruct the model."""
        return {
            "model": "mf",
            "num_users": self.num_users,
            "num_movies": self.num_movies,
            "embedding_dim": self.embedding_dim,
            "global_mean": float(self.global_mean.item()),
        }
