"""HTTP tests cho POST /recommend/mock."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.recommendations import router
from cinematch.recommendation.group import (
    DEFAULT_MISERY_THRESHOLD,
)


app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_mock_recommendation_returns_requested_top_k() -> None:
    response = client.post(
        "/recommend/mock",
        json={"room_id": 1, "strategy": "average", "top_k": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "1.0"
    assert payload["strategy"] == "average"
    assert len(payload["recommendations"]) == 3
    assert [item["rank"] for item in payload["recommendations"]] == [1, 2, 3]


def test_mock_recommendation_is_deterministic() -> None:
    request = {"room_id": 1, "strategy": "least_misery", "top_k": 5}

    first = client.post("/recommend/mock", json=request)
    second = client.post("/recommend/mock", json=request)

    assert first.status_code == 200
    assert first.json() == second.json()


def test_average_without_misery_removes_low_scoring_movies() -> None:
    response = client.post(
        "/recommend/mock",
        json={
            "room_id": 1,
            "strategy": "average_without_misery",
            "top_k": 20,
        },
    )

    assert response.status_code == 200
    recommendations = response.json()["recommendations"]

    assert recommendations
    assert all(
        not item["misery_warning"]
        for item in recommendations
    )
    assert all(
        item["minimum_score"] >= DEFAULT_MISERY_THRESHOLD
        for item in recommendations
    )
    assert 5 not in {
        item["movie_id"]
        for item in recommendations
    }


def test_mock_recommendation_rejects_invalid_request() -> None:
    response = client.post(
        "/recommend/mock",
        json={"room_id": 0, "strategy": "median", "top_k": 50},
    )

    assert response.status_code == 422
