"""Tests for scoring a room with the trained model instead of fixed data."""

from __future__ import annotations

import pytest

from app.core.config import settings
from app.services.predictor_service import (
    MemberProfileTooThin,
    RecommenderUnavailable,
    load_recommender,
    recommender_status,
    reset_recommender,
)


@pytest.fixture(scope="module")
def recommender():
    return load_recommender()


def test_serving_bundle_is_available_in_the_repository() -> None:
    status = recommender_status()

    assert status["available"] is True
    assert status["model"] == "mf"


def test_known_ids_drop_movies_outside_the_model(recommender) -> None:
    known = recommender.known_movielens_ids([1, 50, 999_999])

    assert known == [1, 50]


def test_scores_stay_on_the_rating_scale_and_are_reproducible(
    recommender,
) -> None:
    member_ratings = [
        {1: 5.0, 50: 4.0},
        {181: 2.0, 100: 5.0},
    ]
    candidates = [7, 56, 98, 174, 258]

    first = recommender.score_members(member_ratings, candidates)
    second = recommender.score_members(member_ratings, candidates)

    assert first == second
    assert set(first) == set(candidates)
    for scores in first.values():
        assert len(scores) == 2
        assert all(1.0 <= score <= 5.0 for score in scores)


def test_member_without_usable_rating_is_reported(recommender) -> None:
    with pytest.raises(MemberProfileTooThin) as error:
        recommender.score_members(
            [{1: 5.0}, {999_999: 5.0}],
            [7, 56, 98],
        )

    assert "Member 2" in str(error.value)


def test_empty_candidate_list_is_rejected(recommender) -> None:
    with pytest.raises(RecommenderUnavailable):
        recommender.score_members([{1: 5.0}, {50: 4.0}], [999_999])


def test_missing_bundle_is_reported_clearly(monkeypatch) -> None:
    reset_recommender()
    monkeypatch.setattr(
        settings,
        "SERVING_ARTIFACTS_DIR",
        "artifacts/does-not-exist",
    )
    try:
        with pytest.raises(RecommenderUnavailable) as error:
            load_recommender()
        assert "missing" in str(error.value)
        status = recommender_status()
        assert status["available"] is False
    finally:
        reset_recommender()
