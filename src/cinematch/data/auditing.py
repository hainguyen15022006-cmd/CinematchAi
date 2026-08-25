"""Post-split quality audit for recommendation datasets."""

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from cinematch.data.schema import MAPPED_RATING_COLUMNS


class SplitIntegrityError(ValueError):
    """Raised when processed splits violate data integrity."""


@dataclass(frozen=True)
class PartitionProfile:
    """Quality statistics for one interaction partition."""

    rows: int
    users: int
    movies: int
    duplicate_user_movie_rows: int
    rating_distribution: dict[int, int]
    positive_rate: float


@dataclass(frozen=True)
class TemporalSplitAudit:
    """Serializable audit results for three temporal partitions."""

    train: PartitionProfile
    validation: PartitionProfile
    test: PartitionProfile

    total_rows: int

    validation_cold_start_users: tuple[int, ...]
    test_cold_start_users: tuple[int, ...]

    validation_cold_start_movies: tuple[int, ...]
    test_cold_start_movies: tuple[int, ...]

    validation_cold_start_rows: int
    test_cold_start_rows: int

    train_validation_overlap: int
    train_test_overlap: int
    validation_test_overlap: int

    train_validation_temporal_violations: int
    validation_test_temporal_violations: int

    def as_dict(self) -> dict[str, Any]:
        """Convert audit results to a JSON-compatible dictionary."""
        return asdict(self)


def _validate_partition_schema(
    frame: pd.DataFrame,
    partition_name: str,
) -> None:
    """Validate required columns and basic completeness."""
    actual_columns = tuple(frame.columns)

    if actual_columns != MAPPED_RATING_COLUMNS:
        raise SplitIntegrityError(
            f"{partition_name} has unexpected columns: "
            f"expected {MAPPED_RATING_COLUMNS}, "
            f"received {actual_columns}"
        )

    if frame.empty:
        raise SplitIntegrityError(
            f"{partition_name} is empty"
        )

    if frame.isna().any().any():
        raise SplitIntegrityError(
            f"{partition_name} contains missing values"
        )


def _build_partition_profile(
    frame: pd.DataFrame,
    positive_threshold: float,
) -> PartitionProfile:
    """Calculate statistics for one partition."""
    rating_distribution = {
        int(rating): int(count)
        for rating, count
        in (
            frame["rating"]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    return PartitionProfile(
        rows=len(frame),
        users=int(frame["user_id"].nunique()),
        movies=int(frame["movie_id"].nunique()),
        duplicate_user_movie_rows=int(
            frame.duplicated(
                subset=["user_id", "movie_id"],
            ).sum()
        ),
        rating_distribution=rating_distribution,
        positive_rate=float(
            (
                frame["rating"]
                >= positive_threshold
            ).mean()
        ),
    )


def _interaction_keys(
    frame: pd.DataFrame,
) -> set[tuple[int, int]]:
    """Return unique user-movie interaction keys."""
    return {
        (int(user_id), int(movie_id))
        for user_id, movie_id
        in zip(
            frame["user_id"],
            frame["movie_id"],
        )
    }


def _count_temporal_violations(
    older: pd.DataFrame,
    newer: pd.DataFrame,
) -> int:
    """Count users whose older partition crosses the newer one."""
    older_boundaries = (
        older.groupby("user_id")["timestamp"]
        .max()
        .rename("older_max")
    )

    newer_boundaries = (
        newer.groupby("user_id")["timestamp"]
        .min()
        .rename("newer_min")
    )

    boundaries = pd.concat(
        [older_boundaries, newer_boundaries],
        axis=1,
        join="inner",
    )

    return int(
        (
            boundaries["older_max"]
            > boundaries["newer_min"]
        ).sum()
    )


def audit_temporal_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    positive_threshold: float = 4.0,
) -> TemporalSplitAudit:
    """Audit coverage, overlap, cold-start and temporal integrity."""
    partitions = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    for name, frame in partitions.items():
        _validate_partition_schema(frame, name)

    train_users = {
        int(value)
        for value in train["user_id"].unique()
    }
    validation_users = {
        int(value)
        for value in validation["user_id"].unique()
    }
    test_users = {
        int(value)
        for value in test["user_id"].unique()
    }

    train_movies = {
        int(value)
        for value in train["movie_id"].unique()
    }
    validation_movies = {
        int(value)
        for value in validation["movie_id"].unique()
    }
    test_movies = {
        int(value)
        for value in test["movie_id"].unique()
    }

    validation_cold_movies = tuple(
        sorted(validation_movies - train_movies)
    )
    test_cold_movies = tuple(
        sorted(test_movies - train_movies)
    )

    train_keys = _interaction_keys(train)
    validation_keys = _interaction_keys(validation)
    test_keys = _interaction_keys(test)

    return TemporalSplitAudit(
        train=_build_partition_profile(
            train,
            positive_threshold,
        ),
        validation=_build_partition_profile(
            validation,
            positive_threshold,
        ),
        test=_build_partition_profile(
            test,
            positive_threshold,
        ),
        total_rows=len(train) + len(validation) + len(test),
        validation_cold_start_users=tuple(
            sorted(validation_users - train_users)
        ),
        test_cold_start_users=tuple(
            sorted(test_users - train_users)
        ),
        validation_cold_start_movies=validation_cold_movies,
        test_cold_start_movies=test_cold_movies,
        validation_cold_start_rows=int(
            validation["movie_id"]
            .isin(validation_cold_movies)
            .sum()
        ),
        test_cold_start_rows=int(
            test["movie_id"]
            .isin(test_cold_movies)
            .sum()
        ),
        train_validation_overlap=len(
            train_keys & validation_keys
        ),
        train_test_overlap=len(
            train_keys & test_keys
        ),
        validation_test_overlap=len(
            validation_keys & test_keys
        ),
        train_validation_temporal_violations=(
            _count_temporal_violations(
                train,
                validation,
            )
        ),
        validation_test_temporal_violations=(
            _count_temporal_violations(
                validation,
                test,
            )
        ),
    )


def validate_split_integrity(
    audit: TemporalSplitAudit,
    expected_total_rows: int,
) -> None:
    """Reject structural split errors while allowing cold-start items."""
    issues: list[str] = []

    if audit.total_rows != expected_total_rows:
        issues.append(
            f"expected {expected_total_rows} rows, "
            f"received {audit.total_rows}"
        )

    if audit.validation_cold_start_users:
        issues.append(
            "validation contains cold-start users"
        )

    if audit.test_cold_start_users:
        issues.append(
            "test contains cold-start users"
        )

    overlap_total = (
        audit.train_validation_overlap
        + audit.train_test_overlap
        + audit.validation_test_overlap
    )

    if overlap_total > 0:
        issues.append(
            f"found {overlap_total} overlapping interactions"
        )

    violation_total = (
        audit.train_validation_temporal_violations
        + audit.validation_test_temporal_violations
    )

    if violation_total > 0:
        issues.append(
            f"found {violation_total} temporal violations"
        )

    if issues:
        raise SplitIntegrityError("; ".join(issues))
