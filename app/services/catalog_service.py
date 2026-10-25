"""Import the movie catalog into the Backend database.

The catalog is derived data, so it is not committed as a plain CSV. A
deployment ships the compressed copy inside the serving bundle, and a
development machine usually has the full ``data/processed/movies.csv``.
Both are handled here so the API and the CLI seed the same way.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.movie import Movie

GENRE_COLUMNS = (
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
    "unknown",
)
REQUIRED_COLUMNS = {"movie_id", "title", "release_year", "imdb_url"}


def processed_catalog_path() -> Path:
    return settings.project_root / "data" / "processed" / "movies.csv"


def bundled_catalog_path() -> Path:
    base = Path(settings.SERVING_ARTIFACTS_DIR)
    if not base.is_absolute():
        base = settings.project_root / base
    return base / settings.SERVING_MOVIES_FILE


def resolve_catalog_path(preferred: Path | None = None) -> Path:
    """Return the first catalog that exists: argument, processed, bundle."""
    candidates = [
        path
        for path in (
            preferred,
            processed_catalog_path(),
            bundled_catalog_path(),
        )
        if path is not None
    ]
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    return candidates[0].expanduser().resolve()


def _genres(row: "pd.Series", columns: set[str]) -> str | None:
    selected = [
        genre
        for genre in GENRE_COLUMNS
        if genre in columns and int(row.get(genre, 0)) == 1
    ]
    return "|".join(selected) if selected else None


def seed_movies(csv_path: Path, db: Session) -> tuple[int, int]:
    """Insert new movies and update existing ones keyed by MovieLens ID.

    Args:
        csv_path: A ``movies.csv`` or ``movies.csv.gz`` catalog.
        db: Open database session; the function commits it.

    Returns:
        How many rows were inserted and how many were updated.

    Raises:
        FileNotFoundError: If the catalog does not exist.
        ValueError: If the catalog is missing a required column.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Movie catalog not found: {csv_path}. "
            "Run python scripts/prepare_data.py or "
            "python scripts/build_serving_bundle.py first."
        )

    frame = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Movie catalog is missing columns: {sorted(missing)}"
        )

    existing = {movie.movielens_id: movie for movie in db.query(Movie).all()}
    columns = set(frame.columns)
    inserted = 0
    updated = 0

    for _, row in frame.iterrows():
        movielens_id = int(row["movie_id"])
        movie = existing.get(movielens_id)
        if movie is None:
            movie = Movie(movielens_id=movielens_id)
            db.add(movie)
            existing[movielens_id] = movie
            inserted += 1
        else:
            updated += 1

        movie.title = str(row["title"])
        movie.genres = _genres(row, columns)
        movie.release_year = (
            int(row["release_year"])
            if pd.notna(row["release_year"])
            else None
        )
        movie.imdb_url = (
            str(row["imdb_url"]) if pd.notna(row["imdb_url"]) else None
        )

    db.commit()
    return inserted, updated


def seed_movies_if_empty(db: Session) -> int:
    """Seed the catalog only when the database has no movie yet.

    Returns the number of inserted rows, or 0 when nothing was needed or
    no catalog is available. A missing catalog is not an error here: the
    API still starts and reports the problem through /recommend/status.
    """
    if db.query(Movie).first() is not None:
        return 0
    try:
        inserted, _ = seed_movies(resolve_catalog_path(), db)
    except (FileNotFoundError, ValueError):
        return 0
    return inserted
