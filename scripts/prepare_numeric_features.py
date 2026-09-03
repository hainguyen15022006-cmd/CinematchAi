"""Generate leakage-safe numeric features for Hybrid NCF."""

import argparse
import logging
from pathlib import Path

from cinematch.data.configuration import load_data_config
from cinematch.data.io import (
    load_processed_movies,
    load_processed_ratings,
)
from cinematch.data.mapping import load_id_mappings
from cinematch.features.numeric_features import (
    build_numeric_feature_artifacts,
    save_numeric_feature_artifacts,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "cinematch.yaml"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare CineMatch Hybrid numeric features.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the shared CineMatch YAML configuration.",
    )
    return parser.parse_args()


def main() -> int:
    """Fit train-only preprocessing and save numeric artifacts."""
    args = parse_args()
    config = load_data_config(
        args.config,
        project_root=PROJECT_ROOT,
    )
    train = load_processed_ratings(config.paths.train)
    movies = load_processed_movies(config.paths.movies_processed)
    user_mapping, movie_mapping = load_id_mappings(
        config.paths.mappings
    )
    artifacts = build_numeric_feature_artifacts(
        train,
        movies,
        user_mapping,
        movie_mapping,
        data_version=config.data_version,
        feature_contract_version=(
            config.feature_contract_version
        ),
    )
    saved_paths = save_numeric_feature_artifacts(
        artifacts,
        config.paths.movie_numeric_features,
        config.paths.user_genre_profiles,
        config.paths.numeric_feature_preprocessor,
    )
    scaler = artifacts.preprocessor["release_year"]

    LOGGER.info(
        "Numeric feature dimensions: %d (19 genres + 1 year + 19 history)",
        artifacts.preprocessor["numeric_feature_dimensions"],
    )
    LOGGER.info(
        "Train-only year scaler: median=%.4f, mean=%.4f, std=%.4f",
        scaler["median"],
        scaler["mean"],
        scaler["standard_deviation"],
    )
    LOGGER.info(
        "Rows: movie features=%d, user profiles=%d",
        len(artifacts.movie_features),
        len(artifacts.user_profiles),
    )
    LOGGER.info("Saved movie features: %s", saved_paths[0])
    LOGGER.info("Saved user profiles: %s", saved_paths[1])
    LOGGER.info("Saved preprocessor: %s", saved_paths[2])
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    raise SystemExit(main())
