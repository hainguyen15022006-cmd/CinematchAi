# CineMatch Frontend — Dương (Week 1)

React + TypeScript + Vite frontend for CineMatch, following the existing Backend API/Recommendation Contract.

## Completed scope

- React + TypeScript + Vite, running by default at `http://localhost:5173`.
- Register/Login via the real API; the JWT is stored in `localStorage` and automatically added to the `Authorization` header.
- Onboarding fetches movies from `GET /movies`, rates them 1–5 stars and sends `POST /ratings`.
- The Top 10 page calls `POST /recommend/mock` and displays all of `group_score`, `minimum_score`, `disagreement`, `member_scores`, `misery_warning`, `explanations`.
- The group room page uses the real API to create/join a room and toggle ready.
- Includes loading state, error state, success feedback and login-required routes.

## Running the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Backend

The Backend must be running at `http://127.0.0.1:8000` (or create `.env` from `.env.example` and change `VITE_API_BASE_URL`).

Example of running the backend from the repo root:

```bash
uvicorn app.main:app --reload
```

If the database does not yet contain the movie catalog, follow the Backend's `docs/BACKEND_API.md` before testing `/movies` and `/ratings`.

## Suggested demo flow

1. Open `/register` and create an account.
2. After successful registration, the app logs in automatically and navigates to `/onboarding`.
3. Rate at least 5 movies. Each star click calls the real rating API.
4. Open `/recommendations`, choose one of the 3 strategies and click **Generate Top 10**.
5. Check the fairness/explanation fields for each movie.
6. Optionally open `/room` to create a room, join a room or change the ready status.

## API contract the frontend is using

- `POST /auth/register`
- `POST /auth/login`
- `GET /users/me`
- `GET /movies`
- `POST /ratings`
- `POST /recommend/mock`
- `POST /rooms`
- `GET /rooms/{code}`
- `POST /rooms/{code}/join`
- `POST /rooms/{id}/ready`

The `movie_id` sent with ratings/recommendations is the **MovieLens ID**, not the internal database key.

## Build check before a Pull Request

```bash
npm run build
```

Before committing:

```bash
git status
git add frontend
git commit -m "feat: implement frontend week 1 flow"
git push -u origin feature/frontend-duong
```
