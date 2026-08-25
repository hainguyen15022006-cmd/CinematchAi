"""Deterministic per-user temporal splitting for interactions."""

import math
from dataclasses import dataclass

import pandas as pd

from cinematch.data.schema import MAPPED_RATING_COLUMNS


class TemporalSplitError(ValueError):
    """Raised when interactions cannot be split safely."""


@dataclass(frozen=True)
class TemporalSplit:
    """Train, validation and test interaction partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @property
    def total_rows(self) -> int:
        """Return the number of interactions across all partitions."""
        return len(self.train) + len(self.validation) + len(self.test)


def _allocate_counts(
    number_of_rows: int,
    ratios: tuple[float, float, float],
) -> tuple[int, int, int]:
    """Allocate integer counts with the largest-remainder method."""
    exact_counts = [
        number_of_rows * ratio
        for ratio in ratios
    ]
    counts = [
        math.floor(exact_count)
        for exact_count in exact_counts
    ]
    remaining = number_of_rows - sum(counts)

    priority = sorted(
        range(len(ratios)),
        key=lambda index: (
            exact_counts[index] - counts[index],
            -index,
        ),
        reverse=True,
    )

    for index in priority[:remaining]:
        counts[index] += 1

    return counts[0], counts[1], counts[2]


def temporal_split_by_user(
    ratings: pd.DataFrame,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    minimum_interactions: int = 10,
) -> TemporalSplit:
    """Split every user's interactions from oldest to newest.

    Rows are ordered by ``timestamp`` and then ``movie_id``. The
    secondary key makes the result deterministic when two ratings have
    the same timestamp. Partition sizes use the largest-remainder
    method so integer counts remain as close as possible to 80/10/10.

    The function never uses movie release dates and never shuffles
    interactions randomly.
    """
    ratios = (
        train_ratio,
        validation_ratio,
        test_ratio,
    )

    if any(ratio <= 0.0 for ratio in ratios):
        raise TemporalSplitError(
            "All temporal split ratios must be positive"
        )

    if not math.isclose(sum(ratios), 1.0):
        raise TemporalSplitError(
            "Temporal split ratios must sum to 1.0"
        )

    if minimum_interactions < 3:
        raise TemporalSplitError(
            "minimum_interactions must be at least 3"
        )

    missing_columns = sorted(
        set(MAPPED_RATING_COLUMNS) - set(ratings.columns)
    )
    if missing_columns:
        raise TemporalSplitError(
            f"Mapped ratings are missing columns: {missing_columns}"
        )

    if ratings.empty:
        raise TemporalSplitError("Cannot split empty ratings")

    if ratings[list(MAPPED_RATING_COLUMNS)].isna().any().any():
        raise TemporalSplitError(
            "Mapped ratings contain missing values"
        )

    interactions_per_user = ratings.groupby("user_id").size()
    insufficient_users = interactions_per_user.loc[
        interactions_per_user < minimum_interactions
    ]

    if not insufficient_users.empty:
        examples = insufficient_users.head().to_dict()
        raise TemporalSplitError(
            f"{len(insufficient_users)} users have fewer than "
            f"{minimum_interactions} interactions; "
            f"examples: {examples}"
        )

    ordered = ratings.loc[
        :,
        list(MAPPED_RATING_COLUMNS),
    ].copy()
    ordered["_row_id"] = range(len(ordered))
    ordered = ordered.sort_values(
        ["user_id", "timestamp", "movie_id", "_row_id"],
        kind="mergesort",
    )

    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, user_rows in ordered.groupby(
        "user_id",
        sort=False,
    ):
        train_count, validation_count, test_count = (
            _allocate_counts(
                len(user_rows),
                ratios,
            )
        )

        if min(
            train_count,
            validation_count,
            test_count,
        ) == 0:
            raise TemporalSplitError(
                "Split ratios produce an empty partition for "
                f"a user with {len(user_rows)} interactions"
            )

        validation_end = train_count + validation_count

        train_parts.append(
            user_rows.iloc[:train_count]
        )
        validation_parts.append(
            user_rows.iloc[
                train_count:validation_end
            ]
        )
        test_parts.append(
            user_rows.iloc[
                validation_end:
                validation_end + test_count
            ]
        )

    train = pd.concat(train_parts, ignore_index=True)
    validation = pd.concat(
        validation_parts,
        ignore_index=True,
    )
    test = pd.concat(test_parts, ignore_index=True)

    combined_row_ids = pd.concat(
        [
            train["_row_id"],
            validation["_row_id"],
            test["_row_id"],
        ],
        ignore_index=True,
    )

    if (
        len(combined_row_ids) != len(ratings)
        or combined_row_ids.nunique() != len(ratings)
    ):
        raise RuntimeError(
            "Temporal split lost or duplicated interactions"
        )

    output_columns = list(MAPPED_RATING_COLUMNS)

    return TemporalSplit(
        train=train.loc[:, output_columns].copy(),
        validation=validation.loc[:, output_columns].copy(),
        test=test.loc[:, output_columns].copy(),
    )
