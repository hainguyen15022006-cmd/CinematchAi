"""Audit processed CineMatch temporal splits."""

import argparse
import json
import logging
from pathlib import Path

from cinematch.data.auditing import (
    audit_temporal_splits,
    validate_split_integrity,
)
from cinematch.data.configuration import load_data_config
from cinematch.data.io import load_processed_ratings


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "cinematch.yaml"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Audit CineMatch temporal data splits.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the shared CineMatch YAML configuration.",
    )
    return parser.parse_args()


def main() -> int:
    """Audit generated train, validation and test datasets."""
    args = parse_args()
    config = load_data_config(
        args.config,
        project_root=PROJECT_ROOT,
    )
    train = load_processed_ratings(
        config.paths.train
    )
    validation = load_processed_ratings(
        config.paths.validation
    )
    test = load_processed_ratings(
        config.paths.test
    )

    audit = audit_temporal_splits(
        train,
        validation,
        test,
        positive_threshold=(
            config.positive_rating_threshold
        ),
    )

    validate_split_integrity(
        audit,
        expected_total_rows=config.expected_ratings,
    )

    LOGGER.info(
        "Rows: train=%d, validation=%d, test=%d",
        audit.train.rows,
        audit.validation.rows,
        audit.test.rows,
    )
    LOGGER.info(
        "Users: train=%d, validation=%d, test=%d",
        audit.train.users,
        audit.validation.users,
        audit.test.users,
    )
    LOGGER.info(
        "Movies: train=%d, validation=%d, test=%d",
        audit.train.movies,
        audit.validation.movies,
        audit.test.movies,
    )
    LOGGER.info(
        "Cold-start movies: validation=%d, test=%d",
        len(audit.validation_cold_start_movies),
        len(audit.test_cold_start_movies),
    )
    LOGGER.info(
        "Cold-start rows: validation=%d, test=%d",
        audit.validation_cold_start_rows,
        audit.test_cold_start_rows,
    )
    LOGGER.info(
        "Positive rates: train=%.4f, validation=%.4f, test=%.4f",
        audit.train.positive_rate,
        audit.validation.positive_rate,
        audit.test.positive_rate,
    )
    LOGGER.info(
        "Temporal violations: train-validation=%d, "
        "validation-test=%d",
        audit.train_validation_temporal_violations,
        audit.validation_test_temporal_violations,
    )

    output_path = config.paths.split_audit
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            audit.as_dict(),
            output_file,
            ensure_ascii=False,
            indent=2,
        )
        output_file.write("\n")

    LOGGER.info("Saved audit report: %s", output_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    raise SystemExit(main())
