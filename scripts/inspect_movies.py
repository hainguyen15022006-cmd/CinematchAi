"""Command-line EDA for MovieLens movie metadata."""

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from cinematch.data.io import (
    load_ml100k_movies,
    load_ml100k_ratings,
)
from cinematch.data.profiling import (
    MoviesProfile,
    build_movies_profile,
)
from cinematch.data.validation import (
    validate_movies,
    validate_rating_movie_references,
    validate_ratings,
)


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RATINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ml-100k"
    / "u.data"
)

DEFAULT_MOVIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ml-100k"
    / "u.item"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect MovieLens 100K movie metadata."
        )
    )

    parser.add_argument(
        "--ratings-path",
        type=Path,
        default=DEFAULT_RATINGS_PATH,
    )

    parser.add_argument(
        "--movies-path",
        type=Path,
        default=DEFAULT_MOVIES_PATH,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return parser


def log_profile(profile: MoviesProfile) -> None:
    """Log the main movie metadata statistics."""
    LOGGER.info(
        "Movies: %d",
        profile.number_of_movies,
    )
    LOGGER.info(
        "Duplicate movie ID rows: %d",
        profile.duplicate_movie_id_rows,
    )
    LOGGER.info(
        "Missing titles: %d",
        profile.missing_titles,
    )
    LOGGER.info(
        "Missing release dates: %d",
        profile.missing_release_dates,
    )
    LOGGER.info(
        "Missing IMDb URLs: %d",
        profile.missing_imdb_urls,
    )
    LOGGER.info(
        "Movies without genre: %d",
        profile.movies_without_genre,
    )
    LOGGER.info(
        "Movies with multiple genres: %d",
        profile.movies_with_multiple_genres,
    )
    LOGGER.info(
        "Mean genres per movie: %.4f",
        profile.mean_genres_per_movie,
    )
    LOGGER.info(
        "Release date range: %s -> %s",
        profile.earliest_release_date,
        profile.latest_release_date,
    )
    LOGGER.info(
        "Rated catalog coverage: %.4f%%",
        profile.rated_catalog_coverage * 100,
    )
    LOGGER.info(
        "Missing catalog references: %d",
        profile.missing_catalog_references,
    )
    LOGGER.info(
        "Unreferenced catalog movies: %d",
        profile.unreferenced_catalog_movies,
    )
    LOGGER.info(
        "Genre distribution: %s",
        profile.genre_distribution,
    )


def write_profile(
    profile: MoviesProfile,
    output_path: Path,
) -> None:
    """Write a movie profile as UTF-8 JSON."""
    resolved_output = output_path.resolve()

    resolved_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with resolved_output.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            profile.as_dict(),
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    LOGGER.info(
        "Saved profile: %s",
        resolved_output,
    )


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run movie metadata EDA."""
    arguments = build_parser().parse_args(argv)

    ratings = load_ml100k_ratings(
        arguments.ratings_path
    )

    movies = load_ml100k_movies(
        arguments.movies_path
    )

    validate_ratings(ratings)
    validate_movies(movies)

    validate_rating_movie_references(
        ratings,
        movies,
    )

    profile = build_movies_profile(
        movies,
        ratings,
    )

    log_profile(profile)

    if arguments.output is not None:
        write_profile(
            profile,
            arguments.output,
        )

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    raise SystemExit(main())