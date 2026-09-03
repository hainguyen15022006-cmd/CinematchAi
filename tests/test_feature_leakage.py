"""Tests that Hybrid numeric preprocessing uses train data only."""

import pandas as pd

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
