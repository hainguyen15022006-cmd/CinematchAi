"""Import the processed movie catalog into the Backend database."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import app.models  # noqa: F401 -- register every table with SQLAlchemy
from app.core.db import Base, SessionLocal, engine
from app.services.catalog_service import resolve_catalog_path, seed_movies

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--movies",
        type=Path,
        default=None,
        help=(
            "Catalog to import. Defaults to data/processed/movies.csv, "
            "then artifacts/serving/movies.csv.gz."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    Base.metadata.create_all(bind=engine)
    try:
        with SessionLocal() as db:
            inserted, updated = seed_movies(
                resolve_catalog_path(args.movies), db
            )
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Movies inserted: %d", inserted)
    LOGGER.info("Movies updated: %d", updated)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )
    raise SystemExit(main())
