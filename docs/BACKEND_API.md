# CineMatch Backend API (MVP)

This document describes the week 1 Backend so that Chúc, Dương and the other
members integrate against a single convention. Swagger at
`http://127.0.0.1:8000/docs` is the authoritative reference for the
request/response that is currently running.

## Getting started

```bash
cp .env.example .env
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/seed_movies.py
uvicorn app.main:app --reload
```

`scripts/seed_movies.py` reads `data/processed/movies.csv`. The script can be
run multiple times: new movies are added, and existing movies are updated by
`movielens_id`.

The MVP does not use a migration tool yet. If you already created
`cinematch.db` with the old Backend schema, back it up and then delete that
local database before seeding again. This does not affect committed data
because `.db` files are ignored by Git.

## Authentication

Register or log in to receive a JWT, then send the header:

```http
Authorization: Bearer <access_token>
```

The `/health`, `/movies` and `/recommend/mock` endpoints do not require a token.
The profile, rating, room, run and vote endpoints require a token.

## Main endpoints

| Method | Path | Function |
|---|---|---|
| `GET` | `/health` | Check that the API is running |
| `POST` | `/auth/register` | Create an account |
| `POST` | `/auth/login` | Obtain a JWT |
| `GET` | `/users/me` | Get the current profile |
| `PUT` | `/users/me/preferences` | Update the preference description |
| `GET` | `/movies` | Get the movie catalog |
| `GET` | `/ratings` | Get the saved ratings of the current user |
| `POST` | `/ratings` | Create or update a rating |
| `POST` | `/rooms` | Create a room |
| `GET` | `/rooms/{code}` | Get the room lobby |
| `POST` | `/rooms/{code}/join` | Join a room |
| `POST` | `/rooms/{id}/ready` | Toggle ready status |
| `PUT` | `/rooms/{id}/constraints` | Host updates the constraints |
| `POST` | `/rooms/{id}/recommend` | Host creates a mock run in the DB |
| `GET` | `/runs/{id}/items` | Get the shortlist of a run |
| `POST` | `/runs/{id}/votes` | Create or change the user's choice |
| `POST` | `/runs/{id}/finalize` | Host finalizes the vote result |
| `GET` | `/runs/{id}/result` | Members get the finalized result |
| `POST` | `/recommend/mock` | Mock Top-K following contract v1 |

## ID conventions

- The API accepts and returns `movie_id` as the original MovieLens ID.
- `movies.id` is the internal database key and is not part of the API contract.
- `movie_index` is reserved for the AI model's embeddings.
- The Backend joins AI results with metadata via `movie_id`/`movielens_id`.

## Mock recommendation and the real AI

`POST /recommend/mock` is a database-free endpoint that lets the Frontend
work in parallel while the real AI is not yet finished. The response includes
the full set of fairness fields, described in
`docs/RECOMMENDATION_CONTRACT.md`.

`POST /rooms/{id}/recommend` currently creates a deterministic shortlist from
the movie catalog to test the room/vote flow. When integrating the AI, replace
the adapter in `app/services/ai_service.py`; do not change the agreed response
schema.

## States and voting rules

- A new room is in the `OPEN` state; only the host can run the recommendation.
- A room used for recommendation must have 2 to 5 members; the 6th person is rejected.
- Every member must be `ready` before the host creates a run.
- Only a run in the `VOTING` state accepts votes.
- Each user has exactly one choice per run; submitting again changes the selected movie.
- Only movies in the run's shortlist can be voted for.
- The host calls `finalize` to lock the result; ties are broken by the initial ranking.
- `GET result` does not modify the database and only works after finalization.

## Testing

```bash
python -m pytest -v
```

Backend tests use their own in-memory SQLite and do not delete or modify the
developer's `cinematch.db`.
