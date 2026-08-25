"""Tests for deterministic user and movie ID mappings."""

import json
from pathlib import Path

import pandas as pd
import pytest

from cinematch.data.io import (
    load_ml100k_movies,
    load_ml100k_ratings,
)
from cinematch.data.mapping import (
    UnknownIdentifierError,
    apply_id_mappings,
    build_cinematch_id_mappings,
    build_id_mapping,
    load_id_mappings,
    save_id_mappings,
)
from cinematch.data.schema import MAPPED_RATING_COLUMNS


def test_build_mapping_is_sorted_and_zero_based() -> None:
    mapping = build_id_mapping(
        [20, 5, 20, 9],
        entity_name="movie",
    )

    assert mapping.external_ids == (5, 9, 20)
    assert mapping.external_to_index == {
        5: 0,
        9: 1,
        20: 2,
    }
    assert mapping.size == 3


def test_encode_and_decode_round_trip() -> None:
    mapping = build_id_mapping(
        [5, 9, 20],
        entity_name="movie",
    )
    external_ids = pd.Series(
        [20, 5, 9],
        dtype="int64",
    )

    indices = mapping.encode(external_ids)
    decoded = mapping.decode(indices)

    assert indices.tolist() == [2, 0, 1]
    assert decoded.tolist() == [20, 5, 9]


def test_unknown_external_id_is_rejected() -> None:
    mapping = build_id_mapping(
        [5, 9, 20],
        entity_name="movie",
    )

    with pytest.raises(
        UnknownIdentifierError,
        match="Unknown movie IDs",
    ):
        mapping.encode(
            pd.Series([5, 999], dtype="int64")
        )


def test_invalid_internal_index_is_rejected() -> None:
    mapping = build_id_mapping(
        [5, 9, 20],
        entity_name="movie",
    )

    with pytest.raises(
        UnknownIdentifierError,
        match="Invalid movie indices",
    ):
        mapping.decode(
            pd.Series([0, 3], dtype="int64")
        )


def test_apply_id_mappings_preserves_ratings() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [10, 10, 30],
            "movie_id": [100, 300, 100],
            "rating": [5.0, 3.0, 4.0],
            "timestamp": [1000, 2000, 3000],
        }
    )
    original = ratings.copy(deep=True)
    movies = pd.DataFrame(
        {
            "movie_id": [100, 200, 300],
        }
    )

    user_mapping, movie_mapping = (
        build_cinematch_id_mappings(
            ratings,
            movies,
        )
    )
    mapped = apply_id_mappings(
        ratings,
        user_mapping,
        movie_mapping,
    )

    assert tuple(mapped.columns) == MAPPED_RATING_COLUMNS
    assert len(mapped) == len(ratings)
    assert mapped["user_index"].tolist() == [0, 0, 1]
    assert mapped["movie_index"].tolist() == [0, 2, 0]
    assert mapped["rating"].tolist() == [5.0, 3.0, 4.0]
    pd.testing.assert_frame_equal(ratings, original)


def test_save_and_load_mappings_round_trip(
    tmp_path: Path,
) -> None:
    user_mapping = build_id_mapping(
        [30, 10],
        entity_name="user",
    )
    movie_mapping = build_id_mapping(
        [300, 100, 200],
        entity_name="movie",
    )
    output_path = tmp_path / "id_mappings.json"

    save_id_mappings(
        output_path,
        user_mapping,
        movie_mapping,
    )
    loaded_users, loaded_movies = load_id_mappings(
        output_path
    )

    assert loaded_users == user_mapping
    assert loaded_movies == movie_mapping


def test_unsupported_mapping_version_is_rejected(
    tmp_path: Path,
) -> None:
    mapping_path = tmp_path / "id_mappings.json"
    mapping_path.write_text(
        json.dumps(
            {
                "version": 999,
                "users": {},
                "movies": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported ID mapping version",
    ):
        load_id_mappings(mapping_path)


def test_real_ml100k_mapping_sizes_if_available() -> None:
    project_root = Path(__file__).resolve().parents[1]
    ratings_path = (
        project_root / "data" / "raw" / "ml-100k" / "u.data"
    )
    movies_path = (
        project_root / "data" / "raw" / "ml-100k" / "u.item"
    )

    if not ratings_path.exists() or not movies_path.exists():
        pytest.skip("MovieLens 100K is not downloaded")

    ratings = load_ml100k_ratings(ratings_path)
    movies = load_ml100k_movies(movies_path)
    user_mapping, movie_mapping = (
        build_cinematch_id_mappings(
            ratings,
            movies,
        )
    )
    mapped = apply_id_mappings(
        ratings,
        user_mapping,
        movie_mapping,
    )

    assert user_mapping.size == 943
    assert movie_mapping.size == 1_682
    assert len(mapped) == 100_000
    assert mapped["user_index"].min() == 0
    assert mapped["user_index"].max() == 942
    assert mapped["movie_index"].min() == 0
    assert mapped["movie_index"].max() == 1_681
