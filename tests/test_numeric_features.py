"""Tests for leakage-safe Hybrid numeric feature artifacts."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
import torch

from cinematch.data.io import (
    load_processed_movies,
    load_processed_ratings,
)
from cinematch.data.mapping import IdMapping, load_id_mappings
from cinematch.data.schema import (
    GENRE_COLUMNS,
    MAPPED_RATING_DTYPES,
    PROCESSED_MOVIE_DTYPES,
)
from cinematch.features.numeric_features import (
    HISTORY_COLUMNS,
    NumericFeatureError,
    MOVIE_NUMERIC_COLUMNS,
    NUMERIC_FEATURE_DIM,
    USER_GENRE_PROFILE_COLUMNS,
    build_interaction_numeric_features,
    build_numeric_feature_artifacts,
    fit_release_year_scaler,
    load_numeric_feature_artifacts,
    save_numeric_feature_artifacts,
)
from cinematch.features.hybrid_features import (
    HYBRID_SIDE_FEATURE_DIM,
    build_hybrid_side_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_numeric_feature_fixture():
    """Return train data, catalog and deterministic mappings."""
    movie_data: dict[str, object] = {
        "movie_id": [10, 20, 30, 40],
        "title": ["A", "B", "C", "Future-only"],
        "release_date": [
            "1990-01-01",
            "2000-01-01",
            None,
            "2010-01-01",
        ],
        "release_year": [1990, 2000, None, 2010],
        "release_date_missing": [0, 0, 1, 0],
        "imdb_url": [None, None, None, None],
    }
    for genre in GENRE_COLUMNS:
        movie_data[genre] = [0, 0, 0, 0]
    movie_data["Action"] = [1, 0, 1, 0]
    movie_data["Comedy"] = [0, 1, 1, 0]
    movies = pd.DataFrame(movie_data).astype(PROCESSED_MOVIE_DTYPES)
    movies["release_date"] = pd.to_datetime(movies["release_date"])

    train = pd.DataFrame(
        [
            [1, 10, 0, 0, 5.0, 100],
            [1, 20, 0, 1, 1.0, 110],
            [2, 20, 1, 1, 5.0, 100],
            [2, 30, 1, 2, 3.0, 110],
            [3, 10, 2, 0, 3.0, 100],
        ],
        columns=list(MAPPED_RATING_DTYPES),
    ).astype(MAPPED_RATING_DTYPES)

    return (
        train,
        movies,
        IdMapping("user", (1, 2, 3)),
        IdMapping("movie", (10, 20, 30, 40)),
    )


def build_fixture_artifacts():
    train, movies, user_mapping, movie_mapping = (
        make_numeric_feature_fixture()
    )
    return build_numeric_feature_artifacts(
        train,
        movies,
        user_mapping,
        movie_mapping,
        data_version="fixture-v1",
        feature_contract_version="hybrid-v1-167",
        generated_at=datetime(
            2026,
            9,
            3,
            5,
            0,
            tzinfo=timezone.utc,
        ),
    )


def test_numeric_feature_contract_has_39_columns() -> None:
    artifacts = build_fixture_artifacts()

    assert NUMERIC_FEATURE_DIM == 39
    assert tuple(artifacts.movie_features.columns) == (
        MOVIE_NUMERIC_COLUMNS
    )
    assert tuple(artifacts.user_profiles.columns) == (
        USER_GENRE_PROFILE_COLUMNS
    )
    assert artifacts.preprocessor["numeric_feature_dimensions"] == 39
    assert artifacts.preprocessor["fit_partition"] == "train"


def test_missing_year_uses_train_median_before_scaling() -> None:
    artifacts = build_fixture_artifacts()
    scaler = artifacts.preprocessor["release_year"]
    missing_movie = artifacts.movie_features.loc[
        artifacts.movie_features["movie_id"] == 30
    ].iloc[0]

    assert scaler["median"] == 1995.0
    assert scaler["mean"] == 1995.0
    assert missing_movie["normalized_release_year"] == 0.0
    assert scaler["fit_movie_count"] == 3
    assert scaler["fit_non_missing_count"] == 2


def test_user_genre_history_uses_centered_rating_mean() -> None:
    artifacts = build_fixture_artifacts()
    profiles = artifacts.user_profiles.set_index("user_id")

    assert profiles.loc[1, "history_Action"] == 1.0
    assert profiles.loc[1, "history_Comedy"] == -1.0
    assert profiles.loc[2, "history_Action"] == 0.0
    assert profiles.loc[2, "history_Comedy"] == 0.5
    assert profiles.loc[3, "history_Action"] == 0.0
    assert profiles.loc[3, "history_Romance"] == 0.0


def test_interaction_join_returns_finite_39_column_tensor() -> None:
    train, _, _, _ = make_numeric_feature_fixture()
    artifacts = build_fixture_artifacts()

    tensor = build_interaction_numeric_features(
        train,
        artifacts.movie_features,
        artifacts.user_profiles,
    )

    assert tensor.shape == (len(train), 39)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_numeric_and_text_features_form_167_column_contract() -> None:
    train, _, _, _ = make_numeric_feature_fixture()
    artifacts = build_fixture_artifacts()
    numeric = build_interaction_numeric_features(
        train,
        artifacts.movie_features,
        artifacts.user_profiles,
    )
    text = torch.zeros((len(train), 128), dtype=torch.float32)

    side_features = build_hybrid_side_features(
        numeric[:, :19],
        numeric[:, 19:20],
        numeric[:, 20:39],
        text,
    )

    assert HYBRID_SIDE_FEATURE_DIM == 167
    assert side_features.shape == (len(train), 167)


def test_numeric_artifact_round_trip(tmp_path: Path) -> None:
    artifacts = build_fixture_artifacts()
    movie_path = tmp_path / "movie_features.csv"
    user_path = tmp_path / "user_profiles.csv"
    preprocessor_path = tmp_path / "preprocessor.json"

    save_numeric_feature_artifacts(
        artifacts,
        movie_path,
        user_path,
        preprocessor_path,
    )
    restored = load_numeric_feature_artifacts(
        movie_path,
        user_path,
        preprocessor_path,
    )

    assert len(restored.movie_features) == 4
    assert len(restored.user_profiles) == 3
    assert restored.preprocessor == artifacts.preprocessor


def test_real_movielens_numeric_features_if_available() -> None:
    processed = PROJECT_ROOT / "data" / "processed"
    required_paths = [
        processed / "train.csv",
        processed / "movies.csv",
        processed / "id_mappings.json",
    ]
    if not all(path.is_file() for path in required_paths):
        return

    train = load_processed_ratings(required_paths[0])
    movies = load_processed_movies(required_paths[1])
    user_mapping, movie_mapping = load_id_mappings(required_paths[2])
    artifacts = build_numeric_feature_artifacts(
        train,
        movies,
        user_mapping,
        movie_mapping,
        data_version="ml100k-temporal-v1",
        feature_contract_version="hybrid-v1-167",
    )

    assert len(artifacts.movie_features) == 1682
    assert len(artifacts.user_profiles) == 943
    assert len(HISTORY_COLUMNS) == 19
    numeric_tensor = build_interaction_numeric_features(
        train.head(64),
        artifacts.movie_features,
        artifacts.user_profiles,
    )
    assert numeric_tensor.shape == (64, 39)
    assert torch.isfinite(numeric_tensor).all()


def test_interaction_rejects_unknown_movie_index() -> None:
    artifacts = build_fixture_artifacts()
    interactions = pd.DataFrame(
        [[9, 99, 9, 99, 4.0, 100]],
        columns=list(MAPPED_RATING_DTYPES),
    ).astype(MAPPED_RATING_DTYPES)

    with pytest.raises(NumericFeatureError, match="joined"):
        build_interaction_numeric_features(
            interactions,
            artifacts.movie_features,
            artifacts.user_profiles,
        )


def test_duplicate_movie_id_in_catalog_is_rejected() -> None:
    train, movies, user_mapping, movie_mapping = (
        make_numeric_feature_fixture()
    )
    duplicated = pd.concat(
        [movies, movies.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(NumericFeatureError, match="duplicate movie"):
        build_numeric_feature_artifacts(
            train,
            duplicated,
            user_mapping,
            movie_mapping,
            data_version="fixture-v1",
            feature_contract_version="hybrid-v1-167",
        )


def test_rating_outside_movielens_scale_is_rejected() -> None:
    train, movies, user_mapping, movie_mapping = (
        make_numeric_feature_fixture()
    )
    corrupted = train.copy(deep=True)
    corrupted.loc[0, "rating"] = 6.0

    with pytest.raises(NumericFeatureError, match=r"scale \[1, 5\]"):
        build_numeric_feature_artifacts(
            corrupted,
            movies,
            user_mapping,
            movie_mapping,
            data_version="fixture-v1",
            feature_contract_version="hybrid-v1-167",
        )


def test_nonpositive_release_year_is_rejected() -> None:
    train, movies, _, _ = make_numeric_feature_fixture()
    corrupted = movies.copy(deep=True)
    corrupted.loc[0, "release_year"] = -5

    with pytest.raises(NumericFeatureError, match="positive"):
        fit_release_year_scaler(train, corrupted)
