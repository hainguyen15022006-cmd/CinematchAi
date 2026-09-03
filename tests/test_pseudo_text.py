"""Tests for deterministic train-only pseudo-text features."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cinematch.data.io import (
    load_processed_movies,
    load_processed_ratings,
)
from cinematch.data.mapping import load_id_mappings
from cinematch.features.hybrid_features import (
    HYBRID_SIDE_FEATURE_DIM,
    build_hybrid_side_features,
)
from cinematch.features.numeric_features import (
    build_interaction_numeric_features,
    build_numeric_feature_artifacts,
)
from cinematch.features.pseudo_text import (
    MOVIE_TEXT_COLUMNS,
    TEXT_FEATURE_DIM,
    TEXT_FUSION,
    USER_PSEUDO_TEXT_COLUMNS,
    TextFeatureError,
    build_interaction_text_features,
    build_text_feature_artifacts,
    load_text_feature_artifacts,
    save_text_feature_artifacts,
)
from tests.test_numeric_features import make_numeric_feature_fixture


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_fixture_text_artifacts():
    """Build deterministic artifacts from the shared small fixture."""
    train, movies, user_mapping, movie_mapping = (
        make_numeric_feature_fixture()
    )
    return build_text_feature_artifacts(
        train,
        movies,
        user_mapping,
        movie_mapping,
        data_version="fixture-v1",
        feature_contract_version="hybrid-v1-167",
        positive_rating_threshold=4.0,
        maximum_genres=3,
        seed=42,
        generated_at=datetime(
            2026,
            9,
            3,
            8,
            0,
            tzinfo=timezone.utc,
        ),
    )


def test_pseudo_text_selection_and_fallback_are_explicit() -> None:
    artifacts = build_fixture_text_artifacts()
    users = artifacts.user_texts.set_index("user_id")

    assert tuple(artifacts.user_texts.columns) == (
        USER_PSEUDO_TEXT_COLUMNS
    )
    assert users.loc[1, "preferred_genres"] == "Action"
    assert users.loc[1, "used_fallback"] == np.False_
    assert users.loc[2, "preferred_genres"] == "Comedy"
    assert users.loc[2, "used_fallback"] == np.False_
    assert users.loc[3, "preferred_genres"] == "Action"
    assert users.loc[3, "used_fallback"] == np.True_
    assert users["pseudo_text"].str.endswith(".").all()


def test_minimum_observation_count_filters_small_sample_genres() -> None:
    # User 1 rates one Action movie 5.0 and one Comedy movie 1.0. With a
    # minimum of two observations the single 5-star Action rating no
    # longer qualifies as a preference, so the user falls back instead
    # of getting a genre backed by one lucky rating.
    train, movies, user_mapping, movie_mapping = (
        make_numeric_feature_fixture()
    )
    artifacts = build_text_feature_artifacts(
        train,
        movies,
        user_mapping,
        movie_mapping,
        data_version="fixture-v1",
        feature_contract_version="hybrid-v1-167",
        positive_rating_threshold=4.0,
        maximum_genres=3,
        seed=42,
        minimum_genre_observations=2,
    )
    users = artifacts.user_texts.set_index("user_id")

    assert users.loc[1, "used_fallback"] == np.True_
    assert artifacts.preprocessor[
        "minimum_genre_observations"
    ] == 2


def test_movie_text_uses_only_title_and_genres() -> None:
    artifacts = build_fixture_text_artifacts()
    movies = artifacts.movie_texts.set_index("movie_id")

    assert tuple(artifacts.movie_texts.columns) == MOVIE_TEXT_COLUMNS
    assert movies.loc[10, "movie_text"] == "A. Genres: Action."
    assert movies.loc[30, "movie_text"] == (
        "C. Genres: Action and Comedy."
    )


def test_same_seed_produces_identical_text_and_vectors() -> None:
    first = build_fixture_text_artifacts()
    second = build_fixture_text_artifacts()

    pd.testing.assert_frame_equal(first.user_texts, second.user_texts)
    pd.testing.assert_frame_equal(first.movie_texts, second.movie_texts)
    np.testing.assert_array_equal(first.user_vectors, second.user_vectors)
    np.testing.assert_array_equal(first.movie_vectors, second.movie_vectors)


def test_text_vector_shapes_and_contract_are_valid() -> None:
    artifacts = build_fixture_text_artifacts()

    assert artifacts.user_vectors.shape == (3, TEXT_FEATURE_DIM)
    assert artifacts.movie_vectors.shape == (4, TEXT_FEATURE_DIM)
    assert artifacts.user_vectors.dtype == np.float32
    assert artifacts.movie_vectors.dtype == np.float32
    assert np.isfinite(artifacts.user_vectors).all()
    assert np.isfinite(artifacts.movie_vectors).all()
    assert artifacts.preprocessor["fit_partition"] == "train"
    assert artifacts.preprocessor["text_fusion"] == TEXT_FUSION
    assert artifacts.preprocessor[
        "excluded_user_preference_genres"
    ] == ["unknown"]


def test_text_interaction_keeps_hybrid_contract_at_167() -> None:
    train, movies, user_mapping, movie_mapping = (
        make_numeric_feature_fixture()
    )
    text_artifacts = build_fixture_text_artifacts()
    numeric_artifacts = build_numeric_feature_artifacts(
        train,
        movies,
        user_mapping,
        movie_mapping,
        data_version="fixture-v1",
        feature_contract_version="hybrid-v1-167",
    )
    numeric = build_interaction_numeric_features(
        train,
        numeric_artifacts.movie_features,
        numeric_artifacts.user_profiles,
    )
    text = build_interaction_text_features(train, text_artifacts)

    side_features = build_hybrid_side_features(
        numeric[:, :19],
        numeric[:, 19:20],
        numeric[:, 20:39],
        text,
    )

    assert text.shape == (len(train), 128)
    assert text.dtype == torch.float32
    assert side_features.shape == (
        len(train),
        HYBRID_SIDE_FEATURE_DIM,
    )


def test_future_only_movie_does_not_change_user_pseudo_text() -> None:
    train, movies, user_mapping, movie_mapping = (
        make_numeric_feature_fixture()
    )
    first = build_fixture_text_artifacts()
    changed = movies.copy(deep=True)
    future_only = changed["movie_id"] == 40
    changed.loc[future_only, "Action"] = 1
    changed.loc[future_only, "Romance"] = 1
    second = build_text_feature_artifacts(
        train,
        changed,
        user_mapping,
        movie_mapping,
        data_version="fixture-v1",
        feature_contract_version="hybrid-v1-167",
        positive_rating_threshold=4.0,
        maximum_genres=3,
        seed=42,
    )

    pd.testing.assert_frame_equal(first.user_texts, second.user_texts)
    np.testing.assert_array_equal(first.user_vectors, second.user_vectors)


def test_mapping_drift_is_rejected_for_interaction_features() -> None:
    train, _, _, _ = make_numeric_feature_fixture()
    changed = train.copy(deep=True)
    changed.loc[0, "user_id"] = 2

    with pytest.raises(TextFeatureError, match="mapping drifted"):
        build_interaction_text_features(
            changed,
            build_fixture_text_artifacts(),
        )


def test_text_artifact_round_trip(tmp_path: Path) -> None:
    artifacts = build_fixture_text_artifacts()
    paths = (
        tmp_path / "user_text.csv",
        tmp_path / "movie_text.csv",
        tmp_path / "user_vectors.npz",
        tmp_path / "movie_vectors.npz",
        tmp_path / "preprocessor.json",
    )
    save_text_feature_artifacts(artifacts, *paths)
    restored = load_text_feature_artifacts(*paths)

    pd.testing.assert_frame_equal(restored.user_texts, artifacts.user_texts)
    pd.testing.assert_frame_equal(restored.movie_texts, artifacts.movie_texts)
    np.testing.assert_array_equal(restored.user_vectors, artifacts.user_vectors)
    np.testing.assert_array_equal(
        restored.movie_vectors,
        artifacts.movie_vectors,
    )
    assert restored.preprocessor == artifacts.preprocessor


def test_reordered_user_artifact_is_rejected(tmp_path: Path) -> None:
    artifacts = build_fixture_text_artifacts()
    invalid = replace(
        artifacts,
        user_texts=artifacts.user_texts.iloc[::-1].reset_index(drop=True),
    )

    with pytest.raises(TextFeatureError, match="user_index order"):
        save_text_feature_artifacts(
            invalid,
            tmp_path / "user_text.csv",
            tmp_path / "movie_text.csv",
            tmp_path / "user_vectors.npz",
            tmp_path / "movie_vectors.npz",
            tmp_path / "preprocessor.json",
        )


def test_real_movielens_text_features_if_available() -> None:
    processed = PROJECT_ROOT / "data" / "processed"
    required = (
        processed / "train.csv",
        processed / "movies.csv",
        processed / "id_mappings.json",
    )
    if not all(path.is_file() for path in required):
        pytest.skip("processed MovieLens data is not available")
    train = load_processed_ratings(required[0])
    movies = load_processed_movies(required[1])
    user_mapping, movie_mapping = load_id_mappings(required[2])

    artifacts = build_text_feature_artifacts(
        train,
        movies,
        user_mapping,
        movie_mapping,
        data_version="ml100k-temporal-v1",
        feature_contract_version="hybrid-v1-167",
        positive_rating_threshold=4.0,
        maximum_genres=3,
        seed=42,
    )

    assert artifacts.user_vectors.shape == (943, 128)
    assert artifacts.movie_vectors.shape == (1682, 128)
    assert artifacts.preprocessor["rows"]["fallback_users"] == 85
    assert not artifacts.user_texts["preferred_genres"].str.contains(
        r"(?:^|\|)unknown(?:\||$)",
        regex=True,
    ).any()
    interaction_text = build_interaction_text_features(
        train.head(64),
        artifacts,
    )
    assert interaction_text.shape == (64, 128)
    assert torch.isfinite(interaction_text).all()
