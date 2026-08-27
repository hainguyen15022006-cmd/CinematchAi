"""
test_api_basic.py — Test cơ bản cho auth, movies, ratings.

Áp dụng tư duy Input Domain Modeling (từ môn STQA):
với rating, domain hợp lệ là [1, 5]. Ta test biên: 1, 5 (hợp lệ),
và ngay ngoài biên: 0, 6 (không hợp lệ) — thay vì test ngẫu nhiên.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.db import Base, engine, SessionLocal
from app import models

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_db():
    """Tạo lại DB sạch trước mỗi test, để các test không ảnh hưởng lẫn nhau."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    # seed 1 movie mẫu để test rating
    db = SessionLocal()
    db.add(models.Movie(movielens_id=1, title="Toy Story", genres="Animation"))
    db.commit()
    db.close()
    yield


def register_and_login(email="test@example.com", password="secret123"):
    client.post("/register", json={"email": email, "password": password})
    resp = client.post("/login", json={"email": email, "password": password})
    return resp.json()["access_token"]


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_register_new_user():
    resp = client.post("/register", json={"email": "a@b.com", "password": "secret123"})
    assert resp.status_code == 201
    assert resp.json()["email"] == "a@b.com"
    assert "password" not in resp.json()  # đảm bảo không lộ password


def test_register_duplicate_email_fails():
    client.post("/register", json={"email": "dup@b.com", "password": "secret123"})
    resp = client.post("/register", json={"email": "dup@b.com", "password": "secret123"})
    assert resp.status_code == 400


def test_login_wrong_password_fails():
    client.post("/register", json={"email": "c@b.com", "password": "secret123"})
    resp = client.post("/login", json={"email": "c@b.com", "password": "wrongpass"})
    assert resp.status_code == 401


def test_list_movies():
    resp = client.get("/movies")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_create_rating_without_token_fails():
    resp = client.post("/ratings", json={"movie_id": 1, "rating": 4})
    assert resp.status_code in (401, 422)  # thiếu header Authorization


def test_create_rating_valid_boundary():
    token = register_and_login()
    headers = {"Authorization": f"Bearer {token}"}
    # biên hợp lệ: 1 và 5
    for value in [1, 5]:
        resp = client.post("/ratings", json={"movie_id": 1, "rating": value}, headers=headers)
        assert resp.status_code == 201


def test_create_rating_invalid_boundary():
    token = register_and_login()
    headers = {"Authorization": f"Bearer {token}"}
    # ngay ngoài biên hợp lệ: 0 và 6
    for value in [0, 6]:
        resp = client.post("/ratings", json={"movie_id": 1, "rating": value}, headers=headers)
        assert resp.status_code == 422  # Pydantic tự chặn


def test_recommend_mock():
    resp = client.post("/recommend/mock")
    assert resp.status_code == 200
    assert "top_movies" in resp.json()
