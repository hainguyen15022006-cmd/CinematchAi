"""Dataset used by the NCF smoke-training pipeline."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class NCFDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Store aligned user indices, movie indices and explicit ratings."""

    def __init__(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        ratings: torch.Tensor,
    ) -> None:
        if not all(tensor.ndim == 1 for tensor in (users, items, ratings)):
            raise ValueError("users, items and ratings must be one-dimensional")
        if not (len(users) == len(items) == len(ratings)):
            raise ValueError("users, items and ratings must have equal lengths")
        if len(users) == 0:
            raise ValueError("NCFDataset cannot be empty")
        if torch.any(users < 0) or torch.any(items < 0):
            raise ValueError("embedding indices must be non-negative")
        if not torch.isfinite(ratings).all():
            raise ValueError("ratings must contain only finite values")
        if torch.any((ratings < 1.0) | (ratings > 5.0)):
            raise ValueError("ratings must be within the MovieLens range [1, 5]")

        self.users = users.to(dtype=torch.long)
        self.items = items.to(dtype=torch.long)
        self.ratings = ratings.to(dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return self.users[index], self.items[index], self.ratings[index]
