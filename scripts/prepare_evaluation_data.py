"""Generate shared offline-evaluation inputs for every model."""

import argparse
import logging
from pathlib import Path

from cinematch.data.configuration import load_data_config
from cinematch.data.evaluation_handoff import (
    build_evaluation_handoff,
    save_evaluation_handoff,
)
from cinematch.data.io import load_processed_ratings
from cinematch.data.mapping import load_id_mappings


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "cinematch.yaml"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare shared CineMatch evaluation inputs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the shared CineMatch YAML configuration.",
    )
    return parser.parse_args()


def main() -> int:
    """Build and save the evaluation handoff artifacts."""
    args = parse_args()
    config = load_data_config(
        args.config,
        project_root=PROJECT_ROOT,
    )
    train = load_processed_ratings(config.paths.train)
    validation = load_processed_ratings(config.paths.validation)
    test = load_processed_ratings(config.paths.test)
    user_mapping, movie_mapping = load_id_mappings(
        config.paths.mappings
    )

    handoff = build_evaluation_handoff(
        train,
        validation,
        test,
        user_mapping,
        movie_mapping,
        data_version=config.data_version,
        positive_threshold=config.positive_rating_threshold,
        top_k=config.evaluation_top_k,
        negative_sample_size=config.negative_sample_size,
        random_seed=config.random_seed,
    )
    saved_paths = save_evaluation_handoff(
        handoff,
        config.paths.evaluation_handoff_dir,
    )
    summary = handoff.summary

    LOGGER.info(
        "Users: total=%d, evaluable=%d, skipped=%d",
        summary["users"]["total"],
        summary["users"]["evaluable"],
        summary["users"]["skipped_without_positive"],
    )
    LOGGER.info(
        "Catalog: total=%d, warm-start=%d",
        summary["catalog"]["total_movies"],
        summary["catalog"]["warm_start_movies"],
    )
    LOGGER.info(
        "Cold-start movies: validation=%d, test=%d",
        summary["catalog"]["validation_cold_start_movies"],
        summary["catalog"]["test_cold_start_movies"],
    )
    for filename, path in saved_paths.items():
        LOGGER.info("Saved %s: %s", filename, path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    raise SystemExit(main())
