"""Descriptive profiling for MovieLens ratings."""

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DistributionSummary:
    """Summary statistics for interaction counts."""

    minimum: int
    maximum: int
    mean: float
    median: float


@dataclass(frozen=True)
class RatingsProfile:
    """Serializable EDA result for a ratings dataset."""

    number_of_ratings: int
    number_of_users: int
    number_of_movies: int

    minimum_rating: float
    maximum_rating: float
    mean_rating: float

    missing_values: dict[str, int]
    exact_duplicate_rows: int
    duplicate_user_movie_rows: int

    rating_distribution: dict[int, int]

    ratings_per_user: DistributionSummary
    ratings_per_movie: DistributionSummary

    possible_interactions: int
    density: float
    sparsity: float

    earliest_rating: str
    latest_rating: str

    def as_dict(self) -> dict[str, Any]:
        """Convert the profile into a JSON-serializable dictionary."""
        return asdict(self)


def _summarize_counts(
    counts: pd.Series,
) -> DistributionSummary:
    """Summarize integer interaction counts."""
    return DistributionSummary(
        minimum=int(counts.min()),
        maximum=int(counts.max()),
        mean=float(counts.mean()),
        median=float(counts.median()),
    )


def build_ratings_profile(
    ratings: pd.DataFrame,
) -> RatingsProfile:
    """Calculate reproducible EDA statistics.

    The input is assumed to have passed ``validate_ratings``.
    """
    number_of_ratings = len(ratings)
    number_of_users = int(
        ratings["user_id"].nunique()
    )
    number_of_movies = int(
        ratings["movie_id"].nunique()
    )

    missing_values = {
        str(column): int(count)
        for column, count
        in ratings.isna().sum().items()
    }

    exact_duplicate_rows = int(
        ratings.duplicated().sum()
    )

    duplicate_user_movie_rows = int(
        ratings.duplicated(
            subset=["user_id", "movie_id"],
            keep=False,
        ).sum()
    )

    rating_distribution = {
        int(rating): int(count)
        for rating, count
        in (
            ratings["rating"]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    ratings_per_user = (
        ratings.groupby("user_id")
        .size()
    )

    ratings_per_movie = (
        ratings.groupby("movie_id")
        .size()
    )

    possible_interactions = (
        number_of_users
        * number_of_movies
    )

    density = (
        number_of_ratings
        / possible_interactions
    )

    timestamps = pd.to_datetime(
        ratings["timestamp"],
        unit="s",
        utc=True,
    )

    return RatingsProfile(
        number_of_ratings=number_of_ratings,
        number_of_users=number_of_users,
        number_of_movies=number_of_movies,
        minimum_rating=float(
            ratings["rating"].min()
        ),
        maximum_rating=float(
            ratings["rating"].max()
        ),
        mean_rating=float(
            ratings["rating"].mean()
        ),
        missing_values=missing_values,
        exact_duplicate_rows=exact_duplicate_rows,
        duplicate_user_movie_rows=(
            duplicate_user_movie_rows
        ),
        rating_distribution=rating_distribution,
        ratings_per_user=_summarize_counts(
            ratings_per_user
        ),
        ratings_per_movie=_summarize_counts(
            ratings_per_movie
        ),
        possible_interactions=possible_interactions,
        density=float(density),
        sparsity=float(1.0 - density),
        earliest_rating=timestamps.min().isoformat(),
        latest_rating=timestamps.max().isoformat(),
    )