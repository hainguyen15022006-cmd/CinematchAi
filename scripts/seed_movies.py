"""Import the processed movie catalog into the Backend database."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

import app.models  # noqa: F401 -- register every table with SQLAlchemy
from app.core.db import Base, SessionLocal, engine
from app.models.movie import Movie


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOVIES = PROJECT_ROOT / "data" / "processed" / "movies.csv"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movies", type=Path, default=DEFAULT_MOVIES)
    return parser.parse_args()


def _genres(row: pd.Series, columns: set[str]) -> str | None:
    selected = [
        genre
        for genre in GENRE_COLUMNS
        if genre in columns and int(row.get(genre, 0)) == 1
    ]
    return "|".join(selected) if selected else None


def seed_movies(csv_path: Path, db: Session) -> tuple[int, int]:
    """Insert new movies and update existing ones keyed by MovieLens ID."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Movie catalog not found: {csv_path}. "
            "Run python scripts/prepare_data.py first."
        )

    frame = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Movie catalog is missing columns: {sorted(missing)}"
        )

    existing = {
        movie.movielens_id: movie
        for movie in db.query(Movie).all()
    }
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
        movie.genres = _genres(row, set(frame.columns))
        movie.release_year = (
            int(row["release_year"])
            if pd.notna(row["release_year"])
            else None
        )
        movie.imdb_url = (
            str(row["imdb_url"])
            if pd.notna(row["imdb_url"])
            else None
        )

    db.commit()
    return inserted, updated


def main() -> int:
    args = parse_args()
    Base.metadata.create_all(bind=engine)
    try:
        with SessionLocal() as db:
            inserted, updated = seed_movies(args.movies.resolve(), db)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Movies inserted: %d", inserted)
    LOGGER.info("Movies updated: %d", updated)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
