"""Descriptive profiling for MovieLens ratings."""

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from cinematch.data.schema import GENRE_COLUMNS


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


@dataclass(frozen=True)
class MoviesProfile:
    """Serializable EDA result for movie metadata."""

    number_of_movies: int
    duplicate_movie_id_rows: int

    missing_titles: int
    missing_release_dates: int
    missing_imdb_urls: int

    movies_without_genre: int
    movies_with_multiple_genres: int
    mean_genres_per_movie: float

    genre_distribution: dict[str, int]

    earliest_release_date: str | None
    latest_release_date: str | None

    rated_movie_count: int
    rated_catalog_coverage: float
    unreferenced_catalog_movies: int
    missing_catalog_references: int

    def as_dict(self) -> dict[str, Any]:
        """Convert the profile into a JSON-compatible dictionary."""
        return asdict(self)


def build_movies_profile(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
) -> MoviesProfile:
    """Calculate EDA statistics for movie metadata."""
    genre_frame = movies[list(GENRE_COLUMNS)]

    genres_per_movie = genre_frame.sum(axis=1)

    genre_distribution = {
        genre: int(count)
        for genre, count
        in (
            genre_frame
            .sum(axis=0)
            .sort_values(ascending=False)
            .items()
        )
    }

    release_dates = pd.to_datetime(
        movies["release_date"],
        format="%d-%b-%Y",
        errors="coerce",
    )

    valid_release_dates = (
        release_dates.dropna()
    )

    earliest_release_date: str | None = None
    latest_release_date: str | None = None

    if not valid_release_dates.empty:
        earliest_release_date = (
            valid_release_dates.min()
            .date()
            .isoformat()
        )

        latest_release_date = (
            valid_release_dates.max()
            .date()
            .isoformat()
        )

    missing_title_mask = (
        movies["title"].isna()
        | movies["title"].str.strip().eq("")
    )

    missing_imdb_mask = (
        movies["imdb_url"].isna()
        | movies["imdb_url"].str.strip().eq("")
    )

    catalog_movie_ids = {
        int(movie_id)
        for movie_id
        in movies["movie_id"].unique()
    }

    rated_movie_ids = {
        int(movie_id)
        for movie_id
        in ratings["movie_id"].unique()
    }

    referenced_movie_ids = (
        catalog_movie_ids
        & rated_movie_ids
    )

    missing_catalog_references = (
        rated_movie_ids
        - catalog_movie_ids
    )

    unreferenced_catalog_movies = (
        catalog_movie_ids
        - rated_movie_ids
    )

    if rated_movie_ids:
        rated_catalog_coverage = (
            len(referenced_movie_ids)
            / len(rated_movie_ids)
        )
    else:
        rated_catalog_coverage = 1.0

    return MoviesProfile(
        number_of_movies=len(movies),
        duplicate_movie_id_rows=int(
            movies.duplicated(
                subset=["movie_id"],
                keep=False,
            ).sum()
        ),
        missing_titles=int(
            missing_title_mask.fillna(True).sum()
        ),
        missing_release_dates=int(
            movies["release_date"].isna().sum()
        ),
        missing_imdb_urls=int(
            missing_imdb_mask.fillna(True).sum()
        ),
        movies_without_genre=int(
            (genres_per_movie == 0).sum()
        ),
        movies_with_multiple_genres=int(
            (genres_per_movie > 1).sum()
        ),
        mean_genres_per_movie=float(
            genres_per_movie.mean()
        ),
        genre_distribution=genre_distribution,
        earliest_release_date=earliest_release_date,
        latest_release_date=latest_release_date,
        rated_movie_count=len(rated_movie_ids),
        rated_catalog_coverage=float(
            rated_catalog_coverage
        ),
        unreferenced_catalog_movies=len(
            unreferenced_catalog_movies
        ),
        missing_catalog_references=len(
            missing_catalog_references
        ),
    )
