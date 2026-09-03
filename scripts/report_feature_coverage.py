"""Report feature coverage and fallback statistics for the Data handoff."""

import argparse
import logging
from pathlib import Path

from cinematch.data.configuration import load_data_config
from cinematch.data.io import (
    load_processed_movies,
    load_processed_ratings,
)
from cinematch.features.coverage_report import (
    build_feature_coverage_report,
    save_feature_coverage_report,
)
from cinematch.features.numeric_features import (
    load_numeric_feature_artifacts,
)
from cinematch.features.pseudo_text import load_text_feature_artifacts


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "cinematch.yaml"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Report CineMatch feature coverage and fallback.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the shared CineMatch YAML configuration.",
    )
    return parser.parse_args()


def main() -> int:
    """Load feature artifacts and save the coverage report."""
    args = parse_args()
    config = load_data_config(args.config, project_root=PROJECT_ROOT)
    train = load_processed_ratings(config.paths.train)
    movies = load_processed_movies(config.paths.movies_processed)
    numeric_artifacts = load_numeric_feature_artifacts(
        config.paths.movie_numeric_features,
        config.paths.user_genre_profiles,
        config.paths.numeric_feature_preprocessor,
    )
    text_artifacts = load_text_feature_artifacts(
        config.paths.user_pseudo_text,
        config.paths.movie_text,
        config.paths.user_text_vectors,
        config.paths.movie_text_vectors,
        config.paths.text_feature_preprocessor,
    )

    report = build_feature_coverage_report(
        train,
        movies,
        numeric_artifacts,
        text_artifacts,
        data_version=config.data_version,
    )
    saved_path = save_feature_coverage_report(
        report,
        config.paths.feature_coverage_report,
    )

    users = report["users"]
    LOGGER.info(
        "Pseudo-text fallback users: %d/%d (%.2f%%)",
        users["pseudo_text_fallback"],
        users["total"],
        100.0 * users["pseudo_text_fallback_share"],
    )
    LOGGER.info(
        "Empty history profiles: %d",
        users["empty_history_profiles"],
    )
    LOGGER.info(
        "Movies missing release year (imputed): %d; not in train: %d",
        report["movies"]["missing_release_year_imputed"],
        report["movies"]["not_in_train"],
    )
    LOGGER.info(
        "Normalized year range: [%.2f, %.2f]; outliers beyond |%.1f|: %d",
        report["normalized_release_year"]["minimum"],
        report["normalized_release_year"]["maximum"],
        report["normalized_release_year"]["warning_limit"],
        report["normalized_release_year"]["outliers_beyond_limit"],
    )
    LOGGER.info(
        "Text vectors unit-norm: %s",
        report["text_vectors"]["all_unit_norm"],
    )
    LOGGER.info("Saved coverage report: %s", saved_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    raise SystemExit(main())
