# Frontend–Backend Integration Testing, Week 1

This document records how the two parts were merged and verified:

- Base Backend: `codex/backend-recommendation-schema`.
- Frontend: `feature/frontend-duong`.
- Test branch: `integration/fe-be-week1`.

The integration branch is used only for joint testing; it does not replace the
personal branches of Chúc or Dương and does not automatically change `main`.

## 1. Relationship between Backend schemas and Frontend types

The Backend is the source of the contract definition:

- Pydantic schemas: `app/schemas/`.
- API documentation: `docs/BACKEND_API.md`.
- Group Recommendation contract: `docs/RECOMMENDATION_CONTRACT.md`.
- OpenAPI while the server is running: `http://127.0.0.1:8000/openapi.json`.

The Frontend maps the contract to TypeScript at:

```text
frontend/src/types/index.ts
```

The Frontend calls endpoints via:

```text
frontend/src/services/api.ts
```

Important convention: `movie_id` in requests/responses is the original
MovieLens ID. `movies.id` is the internal database key and is not used by the
Frontend in place of the MovieLens ID.

## 2. How the integration branch was created

```bash
git fetch origin
git switch codex/backend-recommendation-schema
git switch -c integration/fe-be-week1
git merge --no-ff origin/feature/frontend-duong \
  -m "merge: integrate frontend and backend week 1"
```

The two branches merged automatically with no conflicts.

## 3. Running the Backend

From the repository root directory:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/seed_movies.py
uvicorn app.main:app --reload
```

The Backend runs at `http://127.0.0.1:8000`; Swagger is at
`http://127.0.0.1:8000/docs`.

## 4. Running the Frontend

Open another Terminal:

```bash
cd frontend
npm ci
npm run dev
```

The Frontend runs at `http://localhost:5173` and reads the Backend URL from
`VITE_API_BASE_URL`. The default value is `http://127.0.0.1:8000`.

## 5. Verified flow

| Step on the Frontend | Real request to the Backend | Result |
|---|---|---|
| Log in | `POST /auth/login` | `200`, JWT received |
| Open onboarding | `GET /movies?limit=50&skip=0` | `200`, 50 movies received |
| Rate Toy Story 4 stars | `POST /ratings` | `201`, rating saved |
| Create room | `POST /rooms` | `201`, room code received |
| Click Ready | `POST /rooms/{id}/ready` | `200`, status `READY` |
| Refresh lobby | `GET /rooms/{code}` | `200`, member list received |
| Generate Top 10 | `POST /recommend/mock` | `200`, 10 movies received |

The Top 10 result was displayed successfully:

- `group_score`, `minimum_score`, `disagreement`;
- `member_scores` of the three illustrative members;
- `misery_warning`;
- the `explanations` list.

`POST /recommend/mock` still uses the week 1 fake predicted scores. Auth, movie
catalog, rating, room and ready ran against the real Backend/database.

## 6. Checks to run before merging

Backend:

```bash
python -m pytest -q
```

Frontend:

```bash
cd frontend
npm run build
```

Check results for the integration branch:

```text
Backend: 70 passed
Frontend: TypeScript and Vite production build succeeded
Seed: 1,682 movie rows
```

## 7. Completion criteria for week 1 integration

- No hard-coded tokens in the Frontend.
- When the Backend is stopped, the Frontend must display a connection error.
- Requests requiring login must carry `Authorization: Bearer <JWT>`.
- Ratings are sent using the MovieLens ID and saved to the database.
- TypeScript types must match Pydantic/OpenAPI.
- The Top 10 displays all fairness fields and explanations.
- Both the Backend tests and the Frontend build pass before the Pull Request.
