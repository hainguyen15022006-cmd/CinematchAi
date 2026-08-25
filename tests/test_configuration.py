"""Tests for the shared CineMatch data configuration."""

from pathlib import Path

import pytest
import yaml

from cinematch.data.configuration import (
    ConfigurationError,
    load_data_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "cinematch.yaml"


def load_default_payload() -> dict[str, object]:
    """Load the repository configuration as mutable test data."""
    with DEFAULT_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        payload = yaml.safe_load(config_file)

    assert isinstance(payload, dict)
    return payload


def write_config(
    tmp_path: Path,
    payload: dict[str, object],
) -> Path:
    """Write a temporary YAML configuration under a project root."""
    config_path = tmp_path / "configs" / "cinematch.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def test_default_data_config_is_valid() -> None:
    config = load_data_config(
        DEFAULT_CONFIG_PATH,
        project_root=PROJECT_ROOT,
    )

    assert config.expected_ratings == 100_000
    assert config.expected_users == 943
    assert config.expected_movies == 1_682
    assert config.train_ratio == pytest.approx(0.8)
    assert config.positive_rating_threshold == 4.0
    assert config.paths.train == (
        PROJECT_ROOT / "data" / "processed" / "train.csv"
    )


def test_invalid_split_ratios_are_rejected(
    tmp_path: Path,
) -> None:
    payload = load_default_payload()
    payload["data"]["temporal_split"]["train_ratio"] = 0.7
    config_path = write_config(tmp_path, payload)

    with pytest.raises(
        ConfigurationError,
        match="must sum to 1.0",
    ):
        load_data_config(
            config_path,
            project_root=tmp_path,
        )


def test_output_path_cannot_escape_project_root(
    tmp_path: Path,
) -> None:
    payload = load_default_payload()
    payload["data"]["processed"]["train_path"] = (
        "../outside/train.csv"
    )
    config_path = write_config(tmp_path, payload)

    with pytest.raises(
        ConfigurationError,
        match="escapes the project root",
    ):
        load_data_config(
            config_path,
            project_root=tmp_path,
        )
