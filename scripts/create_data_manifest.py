"""Create a versioned manifest from prepared CineMatch artifacts."""

import argparse
import logging
from pathlib import Path

from cinematch.data.configuration import load_data_config
from cinematch.data.manifest import (
    build_data_manifest,
    save_data_manifest,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "cinematch.yaml"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create the CineMatch data manifest.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the shared CineMatch YAML configuration.",
    )
    return parser.parse_args()


def main() -> int:
    """Build and save the manifest for the current prepared data."""
    args = parse_args()
    config = load_data_config(
        args.config,
        project_root=PROJECT_ROOT,
    )
    manifest = build_data_manifest(config, PROJECT_ROOT)
    output_path = save_data_manifest(
        manifest,
        config.paths.data_manifest,
    )

    LOGGER.info(
        "Dataset: %s (%s)",
        manifest["dataset"]["name"],
        manifest["dataset"]["version"],
    )
    LOGGER.info(
        "Rows: train=%d, validation=%d, test=%d",
        manifest["split"]["rows"]["train"],
        manifest["split"]["rows"]["validation"],
        manifest["split"]["rows"]["test"],
    )
    LOGGER.info(
        "Feature contract: %s (%d dimensions)",
        manifest["feature_contract"]["version"],
        manifest["feature_contract"]["total_dimensions"],
    )
    LOGGER.info("Saved data manifest: %s", output_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    raise SystemExit(main())
