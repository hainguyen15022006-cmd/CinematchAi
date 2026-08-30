"""Integration tests cho các luồng Backend quan trọng của CineMatch."""

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401 -- đăng ký toàn bộ bảng
from app.core.db import Base, get_db
from app.models.movie import Movie, Rating
from app.models.recommendation import Vote
from app.routers import (
    auth_router,
    movies_router,
    recommendations_router,
    rooms_router,
    runs_router,
    users_router,
)


TEST_ENGINE = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=TEST_ENGINE,
)


def override_get_db() -> Iterator[Session]:
    with TestingSessionLocal() as db:
        yield db


api_app = FastAPI()


@api_app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


api_app.include_router(auth_router)
api_app.include_router(users_router)
api_app.include_router(movies_router)
api_app.include_router(recommendations_router)
api_app.include_router(rooms_router)
api_app.include_router(runs_router)
api_app.dependency_overrides[get_db] = override_get_db
client = TestClient(api_app)


@pytest.fixture(autouse=True)
def clean_test_database() -> Iterator[None]:
    """Mỗi test dùng database in-memory, không chạm cinematch.db."""
    Base.metadata.drop_all(bind=TEST_ENGINE)
    Base.metadata.create_all(bind=TEST_ENGINE)
    with TestingSessionLocal() as db:
        db.add_all(
            [
                Movie(
                    movielens_id=1,
                    title="Toy Story (1995)",
                    genres="Animation|Children|Comedy",
                ),
                Movie(
                    movielens_id=50,
                    title="Star Wars (1977)",
                    genres="Action|Adventure|Sci-Fi",
                ),
                Movie(
                    movielens_id=181,
                    title="Return of the Jedi (1983)",
                    genres="Action|Adventure|Sci-Fi",
                ),
            ]
        )
        db.commit()
    yield


def register_and_login(
    email: str = "test@example.com",
    password: str = "secret123",
) -> tuple[int, str]:
    register = client.post(
        "/auth/register",
        json={"email": email, "password": password},
    )
    assert register.status_code == 201
    login = client.post(
        "/auth/login",
        json={"email": email, "password": password},
    )
    assert login.status_code == 200
    return register.json()["id"], login.json()["access_token"]


def authorization(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_register_login_and_profile() -> None:
    user_id, token = register_and_login()

    profile = client.get("/users/me", headers=authorization(token))

    assert profile.status_code == 200
    assert profile.json() == {
        "id": user_id,
        "email": "test@example.com",
        "preferences_text": None,
    }


def test_register_duplicate_email_fails() -> None:
    register_and_login()

    duplicate = client.post(
        "/auth/register",
        json={"email": "test@example.com", "password": "secret123"},
    )

    assert duplicate.status_code == 400


def test_login_wrong_password_fails() -> None:
    register_and_login()

    response = client.post(
        "/auth/login",
        json={"email": "test@example.com", "password": "wrongpass"},
    )

    assert response.status_code == 401


def test_list_movies_returns_movielens_id() -> None:
    response = client.get("/movies")

    assert response.status_code == 200
    assert len(response.json()) == 3
    assert response.json()[0]["movielens_id"] == 1


def test_create_rating_uses_movielens_id_and_updates_existing() -> None:
    user_id, token = register_and_login()
    headers = authorization(token)

    first = client.post(
        "/ratings",
        json={"movie_id": 50, "rating": 1},
        headers=headers,
    )
    updated = client.post(
        "/ratings",
        json={"movie_id": 50, "rating": 5},
        headers=headers,
    )

    assert first.status_code == 201
    assert updated.status_code == 201
    assert updated.json()["movie_id"] == 50
    assert updated.json()["rating"] == 5
    saved = client.get("/ratings", headers=headers)
    assert saved.status_code == 200
    assert saved.json() == [updated.json()]
    with TestingSessionLocal() as db:
        ratings = db.query(Rating).filter(Rating.user_id == user_id).all()
        assert len(ratings) == 1


def test_create_rating_requires_authentication() -> None:
    response = client.post(
        "/ratings",
        json={"movie_id": 1, "rating": 4},
    )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"

    saved = client.get("/ratings")
    assert saved.status_code == 401


@pytest.mark.parametrize("rating", [0, 6])
def test_create_rating_rejects_invalid_boundary(rating: int) -> None:
    _, token = register_and_login()

    response = client.post(
        "/ratings",
        json={"movie_id": 1, "rating": rating},
        headers=authorization(token),
    )

    assert response.status_code == 422


def test_recommend_mock_matches_contract() -> None:
    response = client.post(
        "/recommend/mock",
        json={"room_id": 1, "strategy": "average", "top_k": 3},
    )

    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 3


def test_single_member_room_cannot_start_recommendation() -> None:
    _, host_token = register_and_login("solo-host@example.com")
    headers = authorization(host_token)
    created = client.post("/rooms", json={"name": "Solo"}, headers=headers)
    room = created.json()

    ready = client.post(f"/rooms/{room['id']}/ready", headers=headers)
    recommendation = client.post(
        f"/rooms/{room['id']}/recommend",
        headers=headers,
    )

    assert ready.status_code == 200
    assert recommendation.status_code == 400
    assert "between 2 and 5 members" in recommendation.json()["detail"]


def test_room_rejects_sixth_member() -> None:
    _, host_token = register_and_login("limit-host@example.com")
    created = client.post(
        "/rooms",
        json={"name": "Full room"},
        headers=authorization(host_token),
    )
    room = created.json()

    for member_number in range(1, 5):
        _, token = register_and_login(f"member-{member_number}@example.com")
        joined = client.post(
            f"/rooms/{room['code']}/join",
            headers=authorization(token),
        )
        assert joined.status_code == 200

    _, sixth_token = register_and_login("member-5@example.com")
    rejected = client.post(
        f"/rooms/{room['code']}/join",
        headers=authorization(sixth_token),
    )

    assert rejected.status_code == 409
    assert rejected.json()["detail"] == "Room already has maximum 5 members"


def test_room_ready_recommend_and_single_vote_flow() -> None:
    _, host_token = register_and_login("host@example.com")
    _, member_token = register_and_login("member@example.com")
    host_headers = authorization(host_token)
    member_headers = authorization(member_token)

    created = client.post(
        "/rooms",
        json={"name": "Friday night"},
        headers=host_headers,
    )
    assert created.status_code == 201
    room = created.json()

    joined = client.post(
        f"/rooms/{room['code']}/join",
        headers=member_headers,
    )
    assert joined.status_code == 200
    fetched = client.get(
        f"/rooms/{room['code']}",
        headers=member_headers,
    )
    assert fetched.status_code == 200
    assert len(fetched.json()["members"]) == 2

    assert client.post(
        f"/rooms/{room['id']}/ready", headers=host_headers
    ).status_code == 200
    assert client.post(
        f"/rooms/{room['id']}/ready", headers=member_headers
    ).status_code == 200

    recommendation = client.post(
        f"/rooms/{room['id']}/recommend",
        headers=host_headers,
    )
    assert recommendation.status_code == 201
    run = recommendation.json()
    assert len(run["items"]) == 3
    assert {item["movie_id"] for item in run["items"]} == {1, 50, 181}
    items = client.get(
        f"/runs/{run['id']}/items",
        headers=member_headers,
    )
    assert items.status_code == 200
    assert items.json() == run["items"]

    first_vote = client.post(
        f"/runs/{run['id']}/votes",
        json={"movie_id": 1},
        headers=member_headers,
    )
    changed_vote = client.post(
        f"/runs/{run['id']}/votes",
        json={"movie_id": 50},
        headers=member_headers,
    )
    assert first_vote.status_code == 201
    assert changed_vote.status_code == 201
    assert changed_vote.json()["movie_id"] == 50
    with TestingSessionLocal() as db:
        assert db.query(Vote).count() == 1

    pending_result = client.get(
        f"/runs/{run['id']}/result",
        headers=member_headers,
    )
    assert pending_result.status_code == 409

    member_cannot_finalize = client.post(
        f"/runs/{run['id']}/finalize",
        headers=member_headers,
    )
    assert member_cannot_finalize.status_code == 403

    finalized = client.post(
        f"/runs/{run['id']}/finalize",
        headers=host_headers,
    )
    assert finalized.status_code == 200
    assert finalized.json()["winner_movie"]["movielens_id"] == 50
    assert finalized.json()["disagreement"] is None

    member_result = client.get(
        f"/runs/{run['id']}/result",
        headers=member_headers,
    )
    assert member_result.status_code == 200
    assert member_result.json() == finalized.json()

    vote_after_finish = client.post(
        f"/runs/{run['id']}/votes",
        json={"movie_id": 1},
        headers=member_headers,
    )
    assert vote_after_finish.status_code == 409
