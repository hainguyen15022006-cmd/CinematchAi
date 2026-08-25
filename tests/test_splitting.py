"""Tests for deterministic per-user temporal splitting."""

from pathlib import Path

import pandas as pd
import pytest

from cinematch.data.io import (
    load_ml100k_ratings,
    load_processed_ratings,
)
from cinematch.data.mapping import (
    apply_id_mappings,
    build_id_mapping,
)
from cinematch.data.splitting import (
    TemporalSplitError,
    temporal_split_by_user,
)


def make_mapped_ratings(
    interactions_per_user: int = 20,
) -> pd.DataFrame:
    """Create two users with chronological mapped interactions."""
    rows: list[dict[str, int | float]] = []

    for user_index, user_id in enumerate([10, 30]):
        for position in range(interactions_per_user):
            rows.append(
                {
                    "user_id": user_id,
                    "movie_id": 1000 + user_index * 100 + position,
                    "user_index": user_index,
                    "movie_index": user_index * 100 + position,
                    "rating": float(position % 5 + 1),
                    "timestamp": 10_000 + position,
                }
            )

    return pd.DataFrame(rows)


def test_temporal_split_uses_80_10_10_per_user() -> None:
    ratings = make_mapped_ratings(
        interactions_per_user=20,
    )

    result = temporal_split_by_user(ratings)

    assert len(result.train) == 32
    assert len(result.validation) == 4
    assert len(result.test) == 4
    assert result.total_rows == 40

    assert result.train.groupby("user_id").size().tolist() == [16, 16]
    assert result.validation.groupby("user_id").size().tolist() == [2, 2]
    assert result.test.groupby("user_id").size().tolist() == [2, 2]


def test_temporal_split_preserves_chronological_order() -> None:
    ratings = make_mapped_ratings().sample(
        frac=1.0,
        random_state=42,
    )

    result = temporal_split_by_user(ratings)

    for user_id in ratings["user_id"].unique():
        train = result.train.loc[
            result.train["user_id"] == user_id
        ]
        validation = result.validation.loc[
            result.validation["user_id"] == user_id
        ]
        test = result.test.loc[
            result.test["user_id"] == user_id
        ]

        assert train["timestamp"].max() <= validation["timestamp"].min()
        assert validation["timestamp"].max() <= test["timestamp"].min()


def test_temporal_split_preserves_every_interaction_once() -> None:
    ratings = make_mapped_ratings()

    result = temporal_split_by_user(ratings)

    original_pairs = set(
        zip(ratings["user_id"], ratings["movie_id"])
    )
    train_pairs = set(
        zip(result.train["user_id"], result.train["movie_id"])
    )
    validation_pairs = set(
        zip(
            result.validation["user_id"],
            result.validation["movie_id"],
        )
    )
    test_pairs = set(
        zip(result.test["user_id"], result.test["movie_id"])
    )

    assert train_pairs.isdisjoint(validation_pairs)
    assert train_pairs.isdisjoint(test_pairs)
    assert validation_pairs.isdisjoint(test_pairs)
    assert train_pairs | validation_pairs | test_pairs == original_pairs


def test_temporal_split_rejects_users_with_too_few_rows() -> None:
    ratings = make_mapped_ratings(
        interactions_per_user=9,
    )

    with pytest.raises(
        TemporalSplitError,
        match="fewer than 10 interactions",
    ):
        temporal_split_by_user(ratings)


def test_temporal_split_rejects_invalid_ratios() -> None:
    ratings = make_mapped_ratings()

    with pytest.raises(
        TemporalSplitError,
        match="must sum to 1.0",
    ):
        temporal_split_by_user(
            ratings,
            train_ratio=0.7,
            validation_ratio=0.1,
            test_ratio=0.1,
        )


def test_processed_rating_csv_round_trip(
    tmp_path: Path,
) -> None:
    result = temporal_split_by_user(
        make_mapped_ratings()
    )
    output_path = tmp_path / "train.csv"
    result.train.to_csv(output_path, index=False)

    loaded = load_processed_ratings(output_path)

    assert len(loaded) == len(result.train)
    assert str(loaded["rating"].dtype) == "float32"
    assert str(loaded["user_index"].dtype) == "int64"
    assert str(loaded["movie_index"].dtype) == "int64"


def test_real_ml100k_temporal_split_if_available() -> None:
    project_root = Path(__file__).resolve().parents[1]
    ratings_path = (
        project_root / "data" / "raw" / "ml-100k" / "u.data"
    )

    if not ratings_path.exists():
        pytest.skip("MovieLens 100K is not downloaded")

    ratings = load_ml100k_ratings(ratings_path)
    user_mapping = build_id_mapping(
        ratings["user_id"],
        entity_name="user",
    )
    movie_mapping = build_id_mapping(
        ratings["movie_id"],
        entity_name="movie",
    )
    mapped = apply_id_mappings(
        ratings,
        user_mapping,
        movie_mapping,
    )

    result = temporal_split_by_user(mapped)

    assert result.total_rows == 100_000
    assert result.train["user_id"].nunique() == 943
    assert result.validation["user_id"].nunique() == 943
    assert result.test["user_id"].nunique() == 943
