"""Validation rules for MovieLens rating data."""

import pandas as pd

from cinematch.data.schema import (
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