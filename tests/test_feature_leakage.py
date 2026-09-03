"""Tests that Hybrid feature preprocessing uses train data only."""

from pathlib import Path

import pandas as pd
import pytest

from cinematch.features.numeric_features import (
    build_numeric_feature_artifacts,
)
from tests.test_numeric_features import make_numeric_feature_fixture


def test_future_only_movie_year_does_not_change_train_scaler() -> None:
    train, movies, user_mapping, movie_mapping = (
        make_numeric_feature_fixture()
    )
    first = build_numeric_feature_artifacts(
        train,
        movies,
        user_mapping,
        movie_mapping,
        data_version="fixture-v1",
        feature_contract_version="hybrid-v1-167",
    )
    changed_movies = movies.copy(deep=True)
    changed_movies.loc[
        changed_movies["movie_id"] == 40,
        "release_year",
    ] = 9999
    second = build_numeric_feature_artifacts(
        train,
        changed_movies,
        user_mapping,
        movie_mapping,
        data_version="fixture-v1",
        feature_contract_version="hybrid-v1-167",
    )

    assert first.preprocessor["release_year"] == (
        second.preprocessor["release_year"]
    )


def test_future_only_genres_do_not_change_user_history() -> None:
    train, movies, user_mapping, movie_mapping = (
        make_numeric_feature_fixture()
    )
    first = build_numeric_feature_artifacts(
        train,
        movies,
        user_mapping,
        movie_mapping,
        data_version="fixture-v1",
        feature_contract_version="hybrid-v1-167",
    )
    changed_movies = movies.copy(deep=True)
    future_mask = changed_movies["movie_id"] == 40
    changed_movies.loc[future_mask, "Action"] = 1
    changed_movies.loc[future_mask, "Romance"] = 1
    second = build_numeric_feature_artifacts(
        train,
        changed_movies,
        user_mapping,
        movie_mapping,
        data_version="fixture-v1",
        feature_contract_version="hybrid-v1-167",
    )

    pd.testing.assert_frame_equal(
        first.user_profiles,
        second.user_profiles,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_SCRIPTS = (
    "scripts/prepare_numeric_features.py",
    "scripts/prepare_text_features.py",
)


def test_feature_scripts_never_read_validation_or_test() -> None:
    # The builders only accept a train frame, so the last place leakage
    # could slip in is the script wiring. Reading the script sources is
    # a blunt but effective guard against someone loading the held-out
    # partitions "just to check something".
    for relative_path in FEATURE_SCRIPTS:
        source = (PROJECT_ROOT / relative_path).read_text(
            encoding="utf-8"
        )
        assert "config.paths.train" in source, relative_path
        assert "paths.validation" not in source, relative_path
        assert "paths.test" not in source, relative_path


def test_real_history_profile_matches_train_only_recomputation() -> None:
    # Independent recomputation on the real artifacts: pick the heaviest
    # user and rebuild the 19-dim history profile with plain pandas.
    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    profiles_path = (
        PROJECT_ROOT
        / "outputs"
        / "features"
        / "user_genre_profiles.csv"
    )
    movies_path = PROJECT_ROOT / "data" / "processed" / "movies.csv"
    if not (
        train_path.exists()
        and profiles_path.exists()
        and movies_path.exists()
    ):
        pytest.skip("Processed MovieLens artifacts are not available")

    train = pd.read_csv(train_path)
    movies = pd.read_csv(movies_path)
    profiles = pd.read_csv(profiles_path)
    genre_columns = list(movies.columns[6:])

    user_index = int(train.groupby("user_index").size().idxmax())
    joined = train.loc[train["user_index"] == user_index].merge(
        movies[["movie_id", *genre_columns]],
        on="movie_id",
    )
    centered = (joined["rating"].astype(float) - 3.0) / 2.0
    expected = []
    for genre in genre_columns:
        membership = joined[genre].astype(float)
        count = membership.sum()
        value = (
            float((membership * centered).sum() / count)
            if count > 0
            else 0.0
        )
        expected.append(value)

    row = profiles.loc[profiles["user_index"] == user_index].iloc[0]
    stored = row.drop(labels=["user_id", "user_index"]).astype(float)
    assert stored.to_numpy() == pytest.approx(expected, abs=1e-5)
