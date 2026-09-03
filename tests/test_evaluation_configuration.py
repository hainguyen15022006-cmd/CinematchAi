"""Tests for the single shared evaluation configuration."""

from pathlib import Path

import pytest

from cinematch.evaluation.configuration import load_evaluation_config


def test_project_evaluation_config_is_complete() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "cinematch.yaml"
    config = load_evaluation_config(path)
    assert config.positive_rating_threshold == 4.0
    assert config.top_k == 10
    assert config.negative_sample_size == 100
    assert config.seeds == (42, 43, 44)
    assert config.neighbor_count == 20
    assert config.profile_sizes == (5, 10)


def test_evaluation_config_rejects_duplicate_seeds(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """evaluation:
  positive_rating_threshold: 4
  top_k: 10
  negative_sample_size: 100
  seeds: [42, 42]
  cold_start:
    neighbor_count: 20
    profile_sizes: [5, 10]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must not contain duplicates"):
        load_evaluation_config(path)
