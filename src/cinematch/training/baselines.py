"""Shared data loading, training and metrics for neural baselines."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class ExplicitRatingDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Torch dataset backed by model-ready CineMatch interactions."""

    def __init__(self, ratings: pd.DataFrame) -> None:
        required = {"user_index", "movie_index", "rating"}
        missing = required - set(ratings.columns)
        if missing:
            raise ValueError(f"ratings are missing columns: {sorted(missing)}")
        if ratings.empty:
            raise ValueError("ratings cannot be empty")
        self.users = torch.as_tensor(
            ratings["user_index"].to_numpy(dtype=np.int64),
            dtype=torch.long,
        )
        self.movies = torch.as_tensor(
            ratings["movie_index"].to_numpy(dtype=np.int64),
            dtype=torch.long,
        )
        self.ratings = torch.as_tensor(
            ratings["rating"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return self.users[index], self.movies[index], self.ratings[index]


@dataclass(frozen=True)
class TrainingResult:
    """History and best validation state from one training run."""

    best_epoch: int
    best_validation_rmse: float
    history: tuple[dict[str, float], ...]


def regression_metrics(
    targets: np.ndarray | Iterable[float],
    predictions: np.ndarray | Iterable[float],
    clamp: tuple[float, float] | None = (1.0, 5.0),
) -> dict[str, float]:
    """Calculate MSE, RMSE and MAE for explicit rating predictions."""
    actual = np.asarray(targets, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    if actual.shape != predicted.shape:
        raise ValueError("targets and predictions must have equal shapes")
    if actual.size == 0:
        raise ValueError("metrics require at least one prediction")
    if clamp is not None:
        predicted = np.clip(predicted, clamp[0], clamp[1])
    errors = predicted - actual
    mse = float(np.mean(np.square(errors)))
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(errors))),
    }


def predict_explicit_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect predictions and targets without tracking gradients."""
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for users, movies, ratings in data_loader:
            output = model(users.to(device), movies.to(device))
            predictions.append(output.cpu().numpy())
            targets.append(ratings.numpy())
    return np.concatenate(targets), np.concatenate(predictions)


def evaluate_explicit_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a neural model with clamped rating metrics."""
    targets, predictions = predict_explicit_model(model, data_loader, device)
    return regression_metrics(targets, predictions)


def fit_explicit_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
) -> TrainingResult:
    """Train with MSE and restore the best validation checkpoint."""
    if min(epochs, patience) <= 0:
        raise ValueError("epochs and patience must be positive")

    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_rmse = math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        sum_squared_error = 0.0
        examples = 0
        for users, movies, ratings in train_loader:
            users = users.to(device)
            movies = movies.to(device)
            ratings = ratings.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(users, movies)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            sum_squared_error += float(loss.item()) * len(ratings)
            examples += len(ratings)

        validation = evaluate_explicit_model(
            model, validation_loader, device
        )
        epoch_record = {
            "epoch": float(epoch),
            "train_rmse": math.sqrt(sum_squared_error / examples),
            "validation_rmse": validation["rmse"],
            "validation_mae": validation["mae"],
        }
        history.append(epoch_record)

        if validation["rmse"] < best_rmse:
            best_rmse = validation["rmse"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return TrainingResult(
        best_epoch=best_epoch,
        best_validation_rmse=best_rmse,
        history=tuple(history),
    )
