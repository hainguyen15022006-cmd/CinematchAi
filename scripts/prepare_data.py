"""Prepare validated MovieLens data for downstream components."""

import argparse
import logging
from pathlib import Path

from cinematch.data.auditing import (
    audit_temporal_splits,
    validate_split_integrity,
)
from cinematch.data.configuration import load_data_config
from cinematch.data.dataset import prepare_movie_metadata
from cinematch.data.io import (
    load_ml100k_movies,
    load_ml100k_ratings,
)
from cinematch.data.mapping import (
    apply_id_mappings,
    build_cinematch_id_mappings,
    save_id_mappings,
)
from cinematch.data.splitting import temporal_split_by_user
from cinematch.data.validation import (
    validate_movies,
    validate_rating_movie_references,
    validate_ratings,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "cinematch.yaml"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare the CineMatch training datasets.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the shared CineMatch YAML configuration.",
    )
    return parser.parse_args()


def main() -> int:
    """Validate, transform, split and save model-ready data."""
    args = parse_args()
    config = load_data_config(
        args.config,
        project_root=PROJECT_ROOT,
    )
    ratings = load_ml100k_ratings(
        config.paths.ratings_raw
    )
    raw_movies = load_ml100k_movies(
        config.paths.movies_raw
    )

    validate_ratings(ratings)
    validate_movies(raw_movies)
    validate_rating_movie_references(
        ratings,
        raw_movies,
    )

    actual_counts = {
        "ratings": len(ratings),
        "users": int(ratings["user_id"].nunique()),
        "movies": len(raw_movies),
    }
    expected_counts = {
        "ratings": config.expected_ratings,
        "users": config.expected_users,
        "movies": config.expected_movies,
    }
    if actual_counts != expected_counts:
        raise ValueError(
            "Dataset counts do not match configuration: "
            f"expected {expected_counts}, received {actual_counts}"
        )

    processed_movies = prepare_movie_metadata(raw_movies)
    user_mapping, movie_mapping = (
        build_cinematch_id_mappings(
            ratings,
            raw_movies,
        )
    )
    mapped_ratings = apply_id_mappings(
        ratings,
        user_mapping,
        movie_mapping,
    )
    temporal_split = temporal_split_by_user(
        mapped_ratings,
        train_ratio=config.train_ratio,
        validation_ratio=config.validation_ratio,
        test_ratio=config.test_ratio,
        minimum_interactions=(
            config.minimum_interactions_per_user
        ),
    )

    split_audit = audit_temporal_splits(
        temporal_split.train,
        temporal_split.validation,
        temporal_split.test,
        positive_threshold=(
            config.positive_rating_threshold
        ),
    )
    validate_split_integrity(
        split_audit,
        expected_total_rows=config.expected_ratings,
    )

    output_path = config.paths.movies_processed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_movies.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    mappings_output = config.paths.mappings
    save_id_mappings(
        mappings_output,
        user_mapping,
        movie_mapping,
    )

    split_outputs = (
        ("train", temporal_split.train, config.paths.train),
        (
            "validation",
            temporal_split.validation,
            config.paths.validation,
        ),
        ("test", temporal_split.test, config.paths.test),
    )

    for split_name, split_frame, split_path in split_outputs:
        split_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        split_frame.to_csv(
            split_path,
            index=False,
        )
        LOGGER.info(
            "Saved %s interactions: %d rows -> %s",
            split_name,
            len(split_frame),
            split_path,
        )

    LOGGER.info("Validated rating rows: %d", len(ratings))
    LOGGER.info("Mapped rating rows: %d", len(mapped_ratings))
    LOGGER.info(
        "Temporal split rows: %d train, %d validation, %d test",
        len(temporal_split.train),
        len(temporal_split.validation),
        len(temporal_split.test),
    )
    LOGGER.info("Raw movie rows: %d", len(raw_movies))
    LOGGER.info("Processed movie rows: %d", len(processed_movies))
    LOGGER.info(
        "Missing release dates retained: %d",
        int(processed_movies["release_date_missing"].sum()),
    )
    LOGGER.info("Saved processed catalog: %s", output_path)
    LOGGER.info("User mapping size: %d", user_mapping.size)
    LOGGER.info("Movie mapping size: %d", movie_mapping.size)
    LOGGER.info("Saved ID mappings: %s", mappings_output)
    LOGGER.info(
        "Post-split integrity: no overlap, no temporal violation; "
        "%d validation and %d test cold-start rows retained",
        split_audit.validation_cold_start_rows,
        split_audit.test_cold_start_rows,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    raise SystemExit(main())
