"""Tests cho hợp đồng dữ liệu Group Recommendation v1."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas.run import (
    AggregationStrategy,
    GroupRecommendationOut,
    RecommendationRequest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_RESPONSE = (
    PROJECT_ROOT / "docs" / "examples" / "recommendation_response.json"
)


def test_recommendation_request_accepts_supported_strategy() -> None:
    request = RecommendationRequest(
        room_id=1,
        strategy="least_misery",
        top_k=10,
    )

    assert request.strategy is AggregationStrategy.LEAST_MISERY
    assert request.top_k == 10


def test_recommendation_request_rejects_unknown_strategy() -> None:
    with pytest.raises(ValidationError):
        RecommendationRequest(room_id=1, strategy="median", top_k=10)


def test_documented_response_matches_schema() -> None:
    payload = json.loads(EXAMPLE_RESPONSE.read_text(encoding="utf-8"))

    response = GroupRecommendationOut.model_validate(payload)

    assert response.schema_version == "1.0"
    assert response.recommendations[0].movie_id == 50
    assert response.recommendations[0].group_score == pytest.approx(4.6)


def test_legacy_names_are_accepted_without_changing_standard_output() -> None:
    payload = {
        "room_id": 1,
        "strategy": "average",
        "top_movies": [
            {
                "movie_id": 50,
                "rank": 1,
                "title": "Star Wars (1977)",
                "score": 4.6,
                "minimum_score": 3.8,
                "disagreement": 0.42,
                "misery_warning": False,
            }
        ],
    }

    response = GroupRecommendationOut.model_validate(payload)
    serialized = response.model_dump()

    assert response.recommendations[0].group_score == pytest.approx(4.6)
    assert "recommendations" in serialized
    assert "top_movies" not in serialized


def test_score_outside_rating_range_is_rejected() -> None:
    payload = json.loads(EXAMPLE_RESPONSE.read_text(encoding="utf-8"))
    payload["recommendations"][0]["group_score"] = 6.0

    with pytest.raises(ValidationError):
        GroupRecommendationOut.model_validate(payload)
