"""Tests for post-split data quality auditing."""

import pandas as pd
import pytest

from cinematch.data.auditing import (
    SplitIntegrityError,
    audit_temporal_splits,
    validate_split_integrity,
)
from cinematch.data.schema import (
    MAPPED_RATING_COLUMNS,
    MAPPED_RATING_DTYPES,
)


def make_partition(
    rows: list[tuple[int, int, int, int, float, int]],
) -> pd.DataFrame:
    """Create a processed interaction partition."""
    return pd.DataFrame(
        rows,
        columns=list(MAPPED_RATING_COLUMNS),
    ).astype(MAPPED_RATING_DTYPES)


def make_valid_splits() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create small chronological train, validation and test sets."""
    train = make_partition(
        [
            (1, 10, 0, 0, 5.0, 100),
            (1, 20, 0, 1, 4.0, 200),
            (2, 10, 1, 0, 3.0, 100),
            (2, 30, 1, 2, 4.0, 200),
        ]
    )

    validation = make_partition(
        [
            (1, 30, 0, 2, 4.0, 300),
            (2, 40, 1, 3, 3.0, 300),
        ]
    )

    test = make_partition(
        [
            (1, 40, 0, 3, 5.0, 400),
            (2, 50, 1, 4, 2.0, 400),
        ]
    )

    return train, validation, test


def test_valid_split_passes_integrity_validation() -> None:
    train, validation, test = make_valid_splits()

    audit = audit_temporal_splits(
        train,
        validation,
        test,
    )

    validate_split_integrity(
        audit,
        expected_total_rows=8,
    )

    assert audit.total_rows == 8
    assert audit.train.users == 2
    assert audit.validation.users == 2
    assert audit.test.users == 2

    assert audit.validation_cold_start_users == ()
    assert audit.test_cold_start_users == ()

    assert audit.validation_cold_start_movies == (40,)
    assert audit.test_cold_start_movies == (40, 50)

    assert audit.validation_cold_start_rows == 1
    assert audit.test_cold_start_rows == 2


def test_positive_threshold_is_configurable() -> None:
    train, validation, test = make_valid_splits()

    audit = audit_temporal_splits(
        train,
        validation,
        test,
        positive_threshold=5.0,
    )

    assert audit.train.positive_rate == pytest.approx(0.25)


def test_overlapping_interaction_is_rejected() -> None:
    train, validation, test = make_valid_splits()

    validation.loc[0, "movie_id"] = 20

    audit = audit_temporal_splits(
        train,
        validation,
        test,
    )

    assert audit.train_validation_overlap == 1

    with pytest.raises(
        SplitIntegrityError,
        match="overlapping interactions",
    ):
        validate_split_integrity(
            audit,
            expected_total_rows=8,
        )


def test_temporal_violation_is_rejected() -> None:
    train, validation, test = make_valid_splits()

    validation.loc[0, "timestamp"] = 150

    audit = audit_temporal_splits(
        train,
        validation,
        test,
    )

    assert (
        audit.train_validation_temporal_violations
        == 1
    )

    with pytest.raises(
        SplitIntegrityError,
        match="temporal violations",
    ):
        validate_split_integrity(
            audit,
            expected_total_rows=8,
        )


def test_cold_start_user_is_rejected() -> None:
    train, validation, test = make_valid_splits()

    validation.loc[0, "user_id"] = 999
    validation.loc[0, "user_index"] = 999

    audit = audit_temporal_splits(
        train,
        validation,
        test,
    )

    assert audit.validation_cold_start_users == (999,)

    with pytest.raises(
        SplitIntegrityError,
        match="cold-start users",
    ):
        validate_split_integrity(
            audit,
            expected_total_rows=8,
        )


def test_wrong_total_row_count_is_rejected() -> None:
    train, validation, test = make_valid_splits()

    audit = audit_temporal_splits(
        train,
        validation,
        test,
    )

    with pytest.raises(
        SplitIntegrityError,
        match="expected 100 rows",
    ):
        validate_split_integrity(
            audit,
            expected_total_rows=100,
        )


def test_missing_required_column_is_rejected() -> None:
    train, validation, test = make_valid_splits()
    test = test.drop(columns=["movie_index"])

    with pytest.raises(
        SplitIntegrityError,
        match="unexpected columns",
    ):
        audit_temporal_splits(
            train,
            validation,
            test,
        )
        
