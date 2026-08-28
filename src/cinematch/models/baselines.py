"""Non-neural baselines for recommendation experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"user_index", "movie_index", "rating"}


@dataclass(frozen=True)
class PopularItem:
    """One item returned by :meth:`MostPopular.recommend`."""

    movie_index: int
    score: float
    rating_count: int


class MostPopular:
    """Bayesian-smoothed popularity baseline fitted on training data.

    The model predicts the same item score for every user. This is
    intentional: it provides a non-personalized lower-bound baseline.
    """

    def __init__(self, prior_count: float = 20.0) -> None:
        if prior_count < 0:
            raise ValueError("prior_count must be non-negative")
        self.prior_count = float(prior_count)
        self.global_mean_: float | None = None
        self.item_scores_: pd.Series | None = None
        self.item_counts_: pd.Series | None = None
        self.seen_by_user_: dict[int, frozenset[int]] = {}

    @property
    def is_fitted(self) -> bool:
        """Return whether training statistics are available."""
        return self.item_scores_ is not None

    def fit(self, ratings: pd.DataFrame) -> "MostPopular":
        """Fit item statistics using one training partition only."""
        missing = REQUIRED_COLUMNS - set(ratings.columns)
        if missing:
            raise ValueError(f"ratings are missing columns: {sorted(missing)}")
        if ratings.empty:
            raise ValueError("ratings cannot be empty")
        if ratings[list(REQUIRED_COLUMNS)].isna().any().any():
            raise ValueError("ratings contain missing values")

        global_mean = float(ratings["rating"].mean())
        grouped = ratings.groupby("movie_index")["rating"].agg(
            ["mean", "count"]
        )
        denominator = grouped["count"] + self.prior_count
        scores = (
            grouped["count"] * grouped["mean"]
            + self.prior_count * global_mean
        ) / denominator

        self.global_mean_ = global_mean
        self.item_scores_ = scores.astype("float64")
        self.item_counts_ = grouped["count"].astype("int64")
        self.seen_by_user_ = {
            int(user): frozenset(int(item) for item in items)
            for user, items in ratings.groupby("user_index")[
                "movie_index"
            ]
        }
        return self

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("MostPopular must be fitted before use")

    def predict(
        self,
        user_indices: np.ndarray | pd.Series | list[int],
        movie_indices: np.ndarray | pd.Series | list[int],
    ) -> np.ndarray:
        """Predict smoothed item scores for aligned user/item pairs."""
        self._check_fitted()
        users = np.asarray(user_indices)
        movies = np.asarray(movie_indices)
        if users.shape != movies.shape:
            raise ValueError("user_indices and movie_indices must align")

        assert self.item_scores_ is not None
        assert self.global_mean_ is not None
        return np.asarray(
            [
                float(self.item_scores_.get(int(item), self.global_mean_))
                for item in movies.reshape(-1)
            ],
            dtype=np.float64,
        ).reshape(movies.shape)

    def recommend(
        self,
        user_index: int,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> list[PopularItem]:
        """Return the highest-scoring training items for one user."""
        self._check_fitted()
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        assert self.item_scores_ is not None
        assert self.item_counts_ is not None
        candidates = self.item_scores_
        if exclude_seen:
            candidates = candidates.drop(
                labels=list(self.seen_by_user_.get(int(user_index), ())),
                errors="ignore",
            )

        ranking = (
            candidates.rename("score")
            .to_frame()
            .assign(rating_count=self.item_counts_)
            .reset_index()
            .sort_values(
                ["score", "rating_count", "movie_index"],
                ascending=[False, False, True],
                kind="mergesort",
            )
            .head(top_k)
        )
        return [
            PopularItem(
                movie_index=int(row.movie_index),
                score=float(row.score),
                rating_count=int(row.rating_count),
            )
            for row in ranking.itertuples(index=False)
        ]
