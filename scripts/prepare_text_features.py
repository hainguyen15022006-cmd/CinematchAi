"""Generate deterministic train-only text features for Hybrid NCF."""

import argparse
import logging
from pathlib import Path

from cinematch.data.configuration import load_data_config
from cinematch.data.io import (
    load_processed_movies,
    load_processed_ratings,
)
from cinematch.data.mapping import load_id_mappings
from cinematch.features.pseudo_text import (
    build_text_feature_artifacts,
    save_text_feature_artifacts,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "cinematch.yaml"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare CineMatch Hybrid text features.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the shared CineMatch YAML configuration.",
    )
    return parser.parse_args()


def main() -> int:
    """Generate source text, vectors and their versioned contract."""
    args = parse_args()
    config = load_data_config(args.config, project_root=PROJECT_ROOT)
    train = load_processed_ratings(config.paths.train)
    movies = load_processed_movies(config.paths.movies_processed)
    user_mapping, movie_mapping = load_id_mappings(
        config.paths.mappings
    )
    artifacts = build_text_feature_artifacts(
        train,
        movies,
        user_mapping,
        movie_mapping,
        data_version=config.data_version,
        feature_contract_version=config.feature_contract_version,
        positive_rating_threshold=config.positive_rating_threshold,
        maximum_genres=config.pseudo_text_maximum_genres,
        minimum_genre_observations=(
            config.pseudo_text_minimum_genre_observations
        ),
        seed=config.random_seed,
        language=config.pseudo_text_language,
    )
    saved_paths = save_text_feature_artifacts(
        artifacts,
        config.paths.user_pseudo_text,
        config.paths.movie_text,
        config.paths.user_text_vectors,
        config.paths.movie_text_vectors,
        config.paths.text_feature_preprocessor,
    )
    rows = artifacts.preprocessor["rows"]
    LOGGER.info(
        "User pseudo-text rows: %d (fallback users: %d)",
        rows["users"],
        rows["fallback_users"],
    )
    LOGGER.info("Movie text rows: %d", rows["movies"])
    LOGGER.info(
        "Vector shapes: users=%s, movies=%s",
        artifacts.user_vectors.shape,
        artifacts.movie_vectors.shape,
    )
    LOGGER.info(
        "Text fusion: %s -> %d dimensions",
        artifacts.preprocessor["text_fusion"],
        artifacts.preprocessor["text_feature_dimensions"],
    )
    for path in saved_paths:
        LOGGER.info("Saved text artifact: %s", path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    raise SystemExit(main())
