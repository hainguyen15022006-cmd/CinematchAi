"""One prediction contract for baselines, NCF and new CineMatch users.

The public contract deliberately uses model indices.  API-facing code must
convert original MovieLens IDs with ``id_mappings.json`` before calling it.
Only training interactions are accepted for nearest-neighbour fold-in so the
new-user path cannot leak validation or test ratings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import torch
from torch import nn

from cinematch.models import (
    GeneralizedMatrixFactorization,
    MatrixFactorization,
    MostPopular,
    NCF,
)


RATING_MIN = 1.0
RATING_MAX = 5.0
REQUIRED_TRAIN_COLUMNS = {"user_index", "movie_index", "rating"}


@runtime_checkable
class Predictor(Protocol):
    """Stable scoring API consumed by the evaluator and Backend adapter."""

    model_name: str

    def score(
        self,
        user_index: int,
        candidates: list[int] | tuple[int, ...] | np.ndarray,
    ) -> np.ndarray:
        """Return one score per candidate, preserving candidate order."""

    def predict_for_new_user(
        self,
        ratings: Mapping[int, float],
        candidates: list[int] | tuple[int, ...] | np.ndarray,
    ) -> np.ndarray:
        """Score candidates for a user absent from the training mapping."""


@dataclass(frozen=True)
class NeighborMatch:
    """One MovieLens neighbour used to fold in a new user."""

    user_index: int
    overlap_count: int
    distance: float
    weight: float


@dataclass(frozen=True)
class NewUserProfile:
    """Auditable result of matching a new user to training users."""

    neighbors: tuple[NeighborMatch, ...]


def _validate_candidates(
    candidates: list[int] | tuple[int, ...] | np.ndarray,
    number_of_movies: int,
) -> np.ndarray:
    values = np.asarray(candidates)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("candidates must be a non-empty one-dimensional list")
    if values.dtype.kind not in "iu":
        raise ValueError("candidates must contain integer movie indices")
    normalized = values.astype(np.int64, copy=False)
    if len(set(normalized.tolist())) != len(normalized):
        raise ValueError("candidates must not contain duplicates")
    if np.any(normalized < 0) or np.any(normalized >= number_of_movies):
        raise ValueError("candidate movie index is outside the model catalog")
    return normalized


def _validate_training_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_TRAIN_COLUMNS - set(ratings.columns)
    if missing:
        raise ValueError(f"training ratings are missing columns: {sorted(missing)}")
    if ratings.empty:
        raise ValueError("training ratings cannot be empty")
    frame = ratings.loc[:, ["user_index", "movie_index", "rating"]].copy()
    if frame.isna().any().any():
        raise ValueError("training ratings contain missing values")
    if frame.duplicated(["user_index", "movie_index"]).any():
        raise ValueError("training ratings contain duplicate user/movie pairs")
    for column in ("user_index", "movie_index"):
        values = frame[column].to_numpy()
        if values.dtype.kind not in "iu" or np.any(values < 0):
            raise ValueError(f"training {column} values must be non-negative integers")
    if not frame["rating"].between(RATING_MIN, RATING_MAX).all():
        raise ValueError("training ratings must be within [1, 5]")
    return frame


class PopularityPredictor:
    """Adapter for the non-personalized Most Popular lower bound."""

    model_name = "most_popular"

    def __init__(self, model: MostPopular, number_of_movies: int) -> None:
        if not model.is_fitted:
            raise ValueError("MostPopular must be fitted before creating a predictor")
        if number_of_movies <= 0:
            raise ValueError("number_of_movies must be positive")
        self.model = model
        self.number_of_movies = int(number_of_movies)

    def score(
        self,
        user_index: int,
        candidates: list[int] | tuple[int, ...] | np.ndarray,
    ) -> np.ndarray:
        movie_indices = _validate_candidates(candidates, self.number_of_movies)
        users = np.full(movie_indices.shape, int(user_index), dtype=np.int64)
        return self.model.predict(users, movie_indices)

    def predict_for_new_user(
        self,
        ratings: Mapping[int, float],
        candidates: list[int] | tuple[int, ...] | np.ndarray,
    ) -> np.ndarray:
        # Popularity is intentionally user-independent; validate supplied ratings
        # only so callers get the same input guarantees as personalized models.
        _validate_new_user_ratings(ratings, self.number_of_movies)
        return self.score(-1, candidates)


def _validate_new_user_ratings(
    ratings: Mapping[int, float],
    number_of_movies: int,
) -> dict[int, float]:
    if not ratings:
        raise ValueError("new user requires at least one rating")
    normalized: dict[int, float] = {}
    for raw_movie, raw_rating in ratings.items():
        if isinstance(raw_movie, bool) or not isinstance(raw_movie, (int, np.integer)):
            raise ValueError("new-user movie indices must be integers")
        movie_index = int(raw_movie)
        if not 0 <= movie_index < number_of_movies:
            raise ValueError("new-user movie index is outside the model catalog")
        if isinstance(raw_rating, bool) or not isinstance(
            raw_rating, (int, float, np.integer, np.floating)
        ):
            raise ValueError("new-user ratings must be numeric")
        rating = float(raw_rating)
        if not np.isfinite(rating) or not RATING_MIN <= rating <= RATING_MAX:
            raise ValueError("new-user ratings must be finite and within [1, 5]")
        normalized[movie_index] = rating
    return normalized


class TorchPredictor:
    """Adapter for MF, GMF and NCF with train-only nearest-neighbour fold-in."""

    def __init__(
        self,
        model: nn.Module,
        training_ratings: pd.DataFrame,
        *,
        model_name: str,
        number_of_users: int,
        number_of_movies: int,
        device: str | torch.device = "cpu",
        neighbor_count: int = 20,
    ) -> None:
        if model_name not in {"mf", "gmf", "ncf"}:
            raise ValueError("model_name must be one of: mf, gmf, ncf")
        if min(number_of_users, number_of_movies, neighbor_count) <= 0:
            raise ValueError("model sizes and neighbor_count must be positive")
        self.model_name = model_name
        self.number_of_users = int(number_of_users)
        self.number_of_movies = int(number_of_movies)
        self.neighbor_count = int(neighbor_count)
        self.device = torch.device(device)
        self.training_ratings = _validate_training_ratings(training_ratings)
        if self.training_ratings["user_index"].max() >= self.number_of_users:
            raise ValueError("training user index is outside the model")
        if self.training_ratings["movie_index"].max() >= self.number_of_movies:
            raise ValueError("training movie index is outside the model")
        self.model = model.to(self.device)
        self.model.eval()

    def score(
        self,
        user_index: int,
        candidates: list[int] | tuple[int, ...] | np.ndarray,
    ) -> np.ndarray:
        if (
            isinstance(user_index, bool)
            or not isinstance(user_index, (int, np.integer))
            or not 0 <= int(user_index) < self.number_of_users
        ):
            raise ValueError("user_index is outside the model")
        movie_indices = _validate_candidates(candidates, self.number_of_movies)
        users = torch.full(
            (len(movie_indices),), int(user_index), dtype=torch.long, device=self.device
        )
        movies = torch.as_tensor(movie_indices, dtype=torch.long, device=self.device)
        with torch.no_grad():
            scores = self.model(users, movies)
        return np.clip(scores.detach().cpu().numpy(), RATING_MIN, RATING_MAX)

    def match_new_user(
        self,
        ratings: Mapping[int, float],
        *,
        exclude_user_indices: frozenset[int] = frozenset(),
    ) -> NewUserProfile:
        """Find deterministic nearest neighbours using co-rated training items."""
        new_ratings = _validate_new_user_ratings(ratings, self.number_of_movies)
        relevant = self.training_ratings[
            self.training_ratings["movie_index"].isin(new_ratings)
        ]
        candidates: list[tuple[int, int, float, float]] = []
        for user_index, rows in relevant.groupby("user_index", sort=True):
            if int(user_index) in exclude_user_indices:
                continue
            expected = np.asarray(
                [new_ratings[int(movie)] for movie in rows["movie_index"]],
                dtype=np.float64,
            )
            observed = rows["rating"].to_numpy(dtype=np.float64)
            distance = float(np.sqrt(np.mean(np.square(expected - observed))))
            overlap = len(rows)
            # More overlap is better; lower RMSE is better. The strictly positive
            # weight also makes an exact match well-defined.
            raw_weight = float(overlap / (1.0 + distance))
            candidates.append((int(user_index), overlap, distance, raw_weight))
        if not candidates:
            raise ValueError("new user has no rated movie in common with training users")
        candidates.sort(key=lambda item: (-item[3], item[2], item[0]))
        selected = candidates[: self.neighbor_count]
        total_weight = sum(item[3] for item in selected)
        neighbors = tuple(
            NeighborMatch(
                user_index=user,
                overlap_count=overlap,
                distance=distance,
                weight=raw_weight / total_weight,
            )
            for user, overlap, distance, raw_weight in selected
        )
        return NewUserProfile(neighbors=neighbors)

    def predict_for_new_user(
        self,
        ratings: Mapping[int, float],
        candidates: list[int] | tuple[int, ...] | np.ndarray,
        *,
        exclude_user_indices: frozenset[int] = frozenset(),
    ) -> np.ndarray:
        movie_indices = _validate_candidates(candidates, self.number_of_movies)
        profile = self.match_new_user(
            ratings, exclude_user_indices=exclude_user_indices
        )
        neighbor_indices = torch.tensor(
            [item.user_index for item in profile.neighbors],
            dtype=torch.long,
            device=self.device,
        )
        weights = torch.tensor(
            [item.weight for item in profile.neighbors],
            dtype=torch.float32,
            device=self.device,
        )
        movies = torch.as_tensor(movie_indices, dtype=torch.long, device=self.device)
        with torch.no_grad():
            scores = self._score_folded_in(neighbor_indices, weights, movies)
        return np.clip(scores.detach().cpu().numpy(), RATING_MIN, RATING_MAX)

    def _score_folded_in(
        self,
        neighbors: torch.Tensor,
        weights: torch.Tensor,
        movies: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.model, MatrixFactorization):
            user_vector = torch.sum(
                self.model.user_embedding(neighbors) * weights[:, None], dim=0
            )
            user_bias = torch.sum(
                self.model.user_bias(neighbors).squeeze(-1) * weights
            )
            item_vectors = self.model.movie_embedding(movies)
            interactions = item_vectors @ user_vector
            item_biases = self.model.movie_bias(movies).squeeze(-1)
            return self.model.global_mean + user_bias + item_biases + interactions
        if isinstance(self.model, GeneralizedMatrixFactorization):
            user_vector = torch.sum(
                self.model.user_embedding(neighbors) * weights[:, None], dim=0
            )
            item_vectors = self.model.movie_embedding(movies)
            interactions = item_vectors * user_vector.unsqueeze(0)
            return self.model.global_mean + self.model.output(interactions).squeeze(-1)
        if isinstance(self.model, NCF):
            user_vector = torch.sum(
                self.model.user_embed(neighbors) * weights[:, None], dim=0
            )
            item_vectors = self.model.item_embed(movies)
            repeated_user = user_vector.unsqueeze(0).expand(len(movies), -1)
            hidden = self.model.mlp(torch.cat([repeated_user, item_vectors], dim=-1))
            raw = self.model.prediction_head(hidden).squeeze(-1)
            return 1.0 + 4.0 * torch.sigmoid(raw)
        raise TypeError(f"Fold-in is not implemented for {type(self.model).__name__}")


def load_torch_predictor(
    checkpoint_path: Path,
    training_ratings: pd.DataFrame,
    *,
    device: str | torch.device = "cpu",
    neighbor_count: int = 20,
) -> TorchPredictor:
    """Load a versioned MF/GMF/NCF checkpoint into the shared predictor."""
    path = checkpoint_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint was not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary")
    config = payload.get("model_config")
    state = payload.get("model_state_dict")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError("checkpoint requires model_config and model_state_dict")
    model_name = config.get("model")
    if model_name == "mf":
        model = MatrixFactorization(
            num_users=int(config["num_users"]),
            num_movies=int(config["num_movies"]),
            embedding_dim=int(config["embedding_dim"]),
            global_mean=float(config["global_mean"]),
        )
        number_of_users = model.num_users
        number_of_movies = model.num_movies
    elif model_name == "gmf":
        model = GeneralizedMatrixFactorization(
            num_users=int(config["num_users"]),
            num_movies=int(config["num_movies"]),
            embedding_dim=int(config["embedding_dim"]),
            global_mean=float(config["global_mean"]),
        )
        number_of_users = model.num_users
        number_of_movies = model.num_movies
    elif model_name == "ncf":
        model = NCF(
            num_users=int(config["num_users"]),
            num_items=int(config["num_movies"]),
            embedding_dim=int(config.get("embedding_dim", 32)),
            layers=tuple(config.get("layers", (64, 32, 16))),
            dropout=float(config.get("dropout", 0.2)),
        )
        number_of_users = int(config["num_users"])
        number_of_movies = int(config["num_movies"])
    else:
        raise ValueError(f"unsupported checkpoint model: {model_name!r}")
    model.load_state_dict(state)
    return TorchPredictor(
        model,
        training_ratings,
        model_name=str(model_name),
        number_of_users=number_of_users,
        number_of_movies=number_of_movies,
        device=device,
        neighbor_count=neighbor_count,
    )
