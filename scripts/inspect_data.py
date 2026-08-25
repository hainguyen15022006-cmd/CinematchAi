"""Command-line EDA for MovieLens 100K ratings."""

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from cinematch.data.io import load_ml100k_ratings
from cinematch.data.profiling import (
    RatingsProfile,
    build_ratings_profile,
)
from cinematch.data.validation import validate_ratings


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RATINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ml-100k"
    / "u.data"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect and profile MovieLens 100K ratings."
        )
    )

    parser.add_argument(
        "--ratings-path",
        type=Path,
        default=DEFAULT_RATINGS_PATH,
        help="Path to the MovieLens u.data file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )

    return parser


def log_profile(profile: RatingsProfile) -> None:
    """Write the main EDA results to the application log."""
    LOGGER.info(
        "Ratings: %d",
        profile.number_of_ratings,
    )
    LOGGER.info(
        "Users: %d",
        profile.number_of_users,
    )
    LOGGER.info(
        "Movies: %d",
        profile.number_of_movies,
    )
    LOGGER.info(
        "Rating range: %.1f - %.1f",
        profile.minimum_rating,
        profile.maximum_rating,
    )
    LOGGER.info(
        "Mean rating: %.4f",
        profile.mean_rating,
    )
    LOGGER.info(
        "Exact duplicate rows: %d",
        profile.exact_duplicate_rows,
    )
    LOGGER.info(
        "Duplicate user-movie rows: %d",
        profile.duplicate_user_movie_rows,
    )
    LOGGER.info(
        "Density: %.4f%%",
        profile.density * 100,
    )
    LOGGER.info(
        "Sparsity: %.4f%%",
        profile.sparsity * 100,
    )
    LOGGER.info(
        "Time range: %s -> %s",
        profile.earliest_rating,
        profile.latest_rating,
    )


def write_profile(
    profile: RatingsProfile,
    output_path: Path,
) -> None:
    """Persist an EDA profile as UTF-8 JSON."""
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
    """Run the MovieLens EDA command."""
    arguments = build_parser().parse_args(argv)

    ratings = load_ml100k_ratings(
        arguments.ratings_path
    )

    validate_ratings(ratings)

    profile = build_ratings_profile(ratings)

    log_profile(profile)

    LOGGER.info(
        "Rating distribution: %s",
        profile.rating_distribution,
    )
    LOGGER.info(
        "Ratings per user: %s",
        profile.ratings_per_user,
    )
    LOGGER.info(
        "Ratings per movie: %s",
        profile.ratings_per_movie,
    )

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