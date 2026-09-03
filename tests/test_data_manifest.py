"""Tests for the reproducible CineMatch data manifest."""

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from cinematch.data.configuration import (
    DataPaths,
    DataPipelineConfig,
)
from cinematch.data.manifest import (
    DataManifestError,
    build_data_manifest,
    save_data_manifest,
)
from cinematch.data.schema import GENRE_COLUMNS


def _write_fixture_artifacts(
    tmp_path: Path,
) -> DataPipelineConfig:
    """Write a small valid temporal dataset and return its config."""
    raw_directory = tmp_path / "data" / "raw" / "ml-100k"
    processed_directory = tmp_path / "data" / "processed"
    raw_directory.mkdir(parents=True)
    processed_directory.mkdir(parents=True)

    ratings_raw = raw_directory / "u.data"
    movies_raw = raw_directory / "u.item"
    ratings_raw.write_text("fixture ratings\n", encoding="utf-8")
    movies_raw.write_text("fixture movies\n", encoding="utf-8")

    columns = [
        "user_id",
        "movie_id",
        "user_index",
        "movie_index",
        "rating",
        "timestamp",
    ]
    partitions = {
        "train.csv": [
            [1, 10, 0, 0, 5.0, 100],
            [2, 20, 1, 1, 4.0, 100],
        ],
        "validation.csv": [
            [1, 20, 0, 1, 4.0, 200],
            [2, 30, 1, 2, 3.0, 200],
        ],
        "test.csv": [
            [1, 30, 0, 2, 5.0, 300],
            [2, 10, 1, 0, 2.0, 300],
        ],
    }
    for filename, rows in partitions.items():
        pd.DataFrame(rows, columns=columns).to_csv(
            processed_directory / filename,
            index=False,
        )

    movie_data: dict[str, object] = {
        "movie_id": [10, 20, 30],
        "title": ["Movie A", "Movie B", "Movie C"],
        "release_date": ["1995-01-01"] * 3,
        "release_year": [1995, 1996, 1997],
        "release_date_missing": [0, 0, 0],
        "imdb_url": [
            "https://example.com/a",
            "https://example.com/b",
            "https://example.com/c",
        ],
    }
    for genre in GENRE_COLUMNS:
        movie_data[genre] = [0, 0, 0]
    movie_data["Action"] = [1, 0, 1]
    movie_data["Comedy"] = [0, 1, 0]
    pd.DataFrame(movie_data).to_csv(
        processed_directory / "movies.csv",
        index=False,
    )

    mappings_path = processed_directory / "id_mappings.json"
    mappings_path.write_text(
        json.dumps(
            {
                "version": 1,
                "users": {"external_ids": [1, 2]},
                "movies": {"external_ids": [10, 20, 30]},
            }
        ),
        encoding="utf-8",
    )

    return DataPipelineConfig(
        random_seed=42,
        dataset_name="fixture",
        data_version="fixture-v1",
        feature_contract_version="hybrid-v1-167",
        feature_contract_dimensions=167,
        expected_ratings=6,
        expected_users=2,
        expected_movies=3,
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
        minimum_interactions_per_user=3,
        positive_rating_threshold=4.0,
        paths=DataPaths(
            ratings_raw=ratings_raw,
            movies_raw=movies_raw,
            train=processed_directory / "train.csv",
            validation=processed_directory / "validation.csv",
            test=processed_directory / "test.csv",
            movies_processed=processed_directory / "movies.csv",
            mappings=mappings_path,
            split_audit=tmp_path / "outputs" / "split_audit.json",
            data_manifest=tmp_path / "outputs" / "data_manifest.json",
        ),
    )


def test_manifest_is_derived_from_artifacts(
    tmp_path: Path,
) -> None:
    config = _write_fixture_artifacts(tmp_path)
    generated_at = datetime(
        2026,
        9,
        3,
        3,
        0,
        tzinfo=timezone.utc,
    )

    manifest = build_data_manifest(
        config,
        tmp_path,
        generated_at=generated_at,
    )

    assert manifest["generated_at_utc"] == "2026-09-03T03:00:00Z"
    assert manifest["dataset"]["version"] == "fixture-v1"
    assert manifest["entities"] == {
        "ratings": 6,
        "users": 2,
        "movies": 3,
        "rated_movies": 3,
    }
    assert manifest["split"]["rows"] == {
        "train": 2,
        "validation": 2,
        "test": 2,
    }
    assert manifest["feature_contract"]["version"] == (
        "hybrid-v1-167"
    )
    assert manifest["feature_contract"]["total_dimensions"] == 167
    assert len(manifest["feature_contract"]["genre_names"]) == 19
    assert len(manifest["artifacts"]["train"]["sha256"]) == 64


def test_manifest_rejects_configured_count_mismatch(
    tmp_path: Path,
) -> None:
    config = _write_fixture_artifacts(tmp_path)
    invalid_config = replace(config, expected_ratings=7)

    with pytest.raises(
        DataManifestError,
        match="counts do not match",
    ):
        build_data_manifest(invalid_config, tmp_path)


def test_manifest_rejects_feature_dimension_mismatch(
    tmp_path: Path,
) -> None:
    config = _write_fixture_artifacts(tmp_path)
    invalid_config = replace(
        config,
        feature_contract_dimensions=168,
    )

    with pytest.raises(
        DataManifestError,
        match="dimensions do not match code",
    ):
        build_data_manifest(invalid_config, tmp_path)


def test_save_data_manifest_writes_readable_json(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "outputs" / "data_manifest.json"
    manifest = {"schema_version": "1.0", "value": 167}

    saved_path = save_data_manifest(manifest, output_path)

    assert saved_path == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == manifest
