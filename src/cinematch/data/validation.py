"""Validation rules for MovieLens rating data."""

import pandas as pd

from cinematch.data.schema import (
    GENRE_COLUMNS,
    MOVIE_COLUMNS,
    RATING_COLUMNS,
    RATING_MAX,
    RATING_MIN,
)


class DataValidationError(ValueError):
    """Raised when input data violates the expected schema."""


def validate_ratings(ratings: pd.DataFrame) -> None:
    """Validate the schema and values of a ratings DataFrame.

    Args:
        ratings:
            MovieLens rating data to validate.

    Raises:
        DataValidationError:
            If one or more validation rules fail.
    """
    issues: list[str] = []

    actual_columns = tuple(ratings.columns)

    if actual_columns != RATING_COLUMNS:
        issues.append(
            "unexpected columns: "
            f"expected {RATING_COLUMNS}, "
            f"received {actual_columns}"
        )

        raise DataValidationError("; ".join(issues))

    if ratings.empty:
        issues.append("ratings data is empty")

    missing_counts = ratings.isna().sum()
    missing_total = int(missing_counts.sum())

    if missing_total > 0:
        issues.append(
            f"found {missing_total} missing values"
        )

    invalid_rating_count = int(
        (
            ~ratings["rating"].between(
                RATING_MIN,
                RATING_MAX,
            )
        ).sum()
    )

    if invalid_rating_count > 0:
        issues.append(
            f"found {invalid_rating_count} ratings "
            f"outside [{RATING_MIN}, {RATING_MAX}]"
        )

    invalid_user_count = int(
        (ratings["user_id"] <= 0).sum()
    )

    if invalid_user_count > 0:
        issues.append(
            f"found {invalid_user_count} non-positive user IDs"
        )

    invalid_movie_count = int(
        (ratings["movie_id"] <= 0).sum()
    )

    if invalid_movie_count > 0:
        issues.append(
            f"found {invalid_movie_count} non-positive movie IDs"
        )

    invalid_timestamp_count = int(
        (ratings["timestamp"] <= 0).sum()
    )

    if invalid_timestamp_count > 0:
        issues.append(
            f"found {invalid_timestamp_count} "
            "non-positive timestamps"
        )

    if issues:
        raise DataValidationError("; ".join(issues))


def validate_movies(movies: pd.DataFrame) -> None:
    """Validate MovieLens movie metadata.

    Args:
        movies:
            MovieLens movie catalog to validate.

    Raises:
        DataValidationError:
            If metadata violates the expected schema.
    """
    issues: list[str] = []

    actual_columns = tuple(movies.columns)

    if actual_columns != MOVIE_COLUMNS:
        raise DataValidationError(
            "unexpected movie columns: "
            f"expected {MOVIE_COLUMNS}, "
            f"received {actual_columns}"
        )

    if movies.empty:
        issues.append("movie catalog is empty")

    missing_movie_ids = int(
        movies["movie_id"].isna().sum()
    )

    if missing_movie_ids > 0:
        issues.append(
            f"found {missing_movie_ids} missing movie IDs"
        )

    invalid_movie_ids = int(
        (movies["movie_id"] <= 0).sum()
    )

    if invalid_movie_ids > 0:
        issues.append(
            f"found {invalid_movie_ids} "
            "non-positive movie IDs"
        )

    duplicate_movie_id_rows = int(
        movies.duplicated(
            subset=["movie_id"],
            keep=False,
        ).sum()
    )

    if duplicate_movie_id_rows > 0:
        issues.append(
            f"found {duplicate_movie_id_rows} rows "
            "with duplicated movie IDs"
        )

    invalid_title_mask = (
        movies["title"].isna()
        | movies["title"].str.strip().eq("")
    )

    invalid_title_count = int(
        invalid_title_mask.fillna(True).sum()
    )

    if invalid_title_count > 0:
        issues.append(
            f"found {invalid_title_count} "
            "missing or blank movie titles"
        )

    genres = movies[list(GENRE_COLUMNS)]

    invalid_genre_value_count = int(
        (~genres.isin([0, 1])).sum().sum()
    )

    if invalid_genre_value_count > 0:
        issues.append(
            f"found {invalid_genre_value_count} "
            "genre values outside {0, 1}"
        )

    if issues:
        raise DataValidationError("; ".join(issues))


def validate_rating_movie_references(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> None:
    """Ensure every rated movie exists in the catalog."""
    rated_movie_ids = {
        int(movie_id)
        for movie_id
        in ratings["movie_id"].unique()
    }

    catalog_movie_ids = {
        int(movie_id)
        for movie_id
        in movies["movie_id"].unique()
    }

    missing_movie_ids = sorted(
        rated_movie_ids - catalog_movie_ids
    )

    if missing_movie_ids:
        examples = missing_movie_ids[:5]

        raise DataValidationError(
            f"{len(missing_movie_ids)} rated movie IDs "
            "are missing from the catalog; "
            f"examples: {examples}"
        )