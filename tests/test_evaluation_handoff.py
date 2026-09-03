"""Tests for Data-to-Evaluation handoff artifacts."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from cinematch.data.evaluation_handoff import (
    EvaluationHandoffError,
    build_evaluation_handoff,
    load_evaluation_handoff,
    save_evaluation_handoff,
)
from cinematch.data.io import load_processed_ratings
from cinematch.data.mapping import IdMapping, load_id_mappings
from cinematch.data.schema import MAPPED_RATING_DTYPES


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _interaction_frame(rows: list[list[object]]) -> pd.DataFrame:
    """Create one mapped interaction partition."""
    return pd.DataFrame(
        rows,
        columns=list(MAPPED_RATING_DTYPES),
    ).astype(MAPPED_RATING_DTYPES)


def _build_fixture_handoff():
    """Build a deterministic two-user handoff fixture."""
    train = _interaction_frame(
        [
            [1, 10, 0, 0, 5.0, 100],
            [2, 20, 1, 1, 4.0, 100],
        ]
    )
    validation = _interaction_frame(
        [
            [1, 20, 0, 1, 4.0, 200],
            [2, 30, 1, 2, 3.0, 200],
        ]
    )
    test = _interaction_frame(
        [
            [1, 30, 0, 2, 5.0, 300],
            [2, 10, 1, 0, 2.0, 300],
        ]
    )
    user_mapping = IdMapping("user", (1, 2))
    movie_mapping = IdMapping("movie", (10, 20, 30))

    return build_evaluation_handoff(
        train,
        validation,
        test,
        user_mapping,
        movie_mapping,
        data_version="fixture-v1",
        positive_threshold=4.0,
        top_k=10,
        negative_sample_size=100,
        random_seed=42,
        generated_at=datetime(
            2026,
            9,
            3,
            4,
            0,
            tzinfo=timezone.utc,
        ),
    )


def test_handoff_uses_train_and_validation_as_seen() -> None:
    handoff = _build_fixture_handoff()

    assert handoff.seen_items["policy"] == "train_union_validation"
    assert handoff.seen_items["users"] == [
        {
            "user_id": 1,
            "user_index": 0,
            "movie_indices": [0, 1],
        },
        {
            "user_id": 2,
            "user_index": 1,
            "movie_indices": [1, 2],
        },
    ]


def test_handoff_keeps_users_without_positive_items() -> None:
    handoff = _build_fixture_handoff()

    assert handoff.positive_test_items["users"] == [
        {
            "user_id": 1,
            "user_index": 0,
            "movie_indices": [2],
        },
        {
            "user_id": 2,
            "user_index": 1,
            "movie_indices": [],
        },
    ]
    assert handoff.summary["users"]["total"] == 2
    assert handoff.summary["users"]["evaluable"] == 1
    assert handoff.summary["users"]["skipped_without_positive"] == 1


def test_handoff_reports_catalog_and_cold_start() -> None:
    handoff = _build_fixture_handoff()

    assert handoff.catalog["number_of_movies"] == 3
    assert handoff.catalog["movies"][2] == {
        "movie_id": 30,
        "movie_index": 2,
        "observed_in_train": False,
    }
    assert handoff.summary["catalog"]["warm_start_movies"] == 2
    assert (
        handoff.summary["catalog"]["validation_cold_start_movies"]
        == 1
    )
    assert handoff.summary["catalog"]["test_cold_start_movies"] == 1


def test_handoff_rejects_inconsistent_id_index_pair() -> None:
    train = _interaction_frame(
        [
            [1, 10, 1, 0, 5.0, 100],
            [2, 20, 1, 1, 4.0, 100],
        ]
    )
    validation = _interaction_frame(
        [
            [1, 20, 0, 1, 4.0, 200],
            [2, 30, 1, 2, 3.0, 200],
        ]
    )
    test = _interaction_frame(
        [
            [1, 30, 0, 2, 5.0, 300],
            [2, 10, 1, 0, 2.0, 300],
        ]
    )

    with pytest.raises(
        EvaluationHandoffError,
        match="inconsistent user IDs and indices",
    ):
        build_evaluation_handoff(
            train,
            validation,
            test,
            IdMapping("user", (1, 2)),
            IdMapping("movie", (10, 20, 30)),
            data_version="fixture-v1",
            positive_threshold=4.0,
            top_k=10,
            negative_sample_size=100,
            random_seed=42,
        )


def test_save_handoff_writes_all_four_documents(
    tmp_path: Path,
) -> None:
    handoff = _build_fixture_handoff()

    saved_paths = save_evaluation_handoff(handoff, tmp_path)

    assert set(saved_paths) == {
        "catalog.json",
        "seen_items.json",
        "positive_test_items.json",
        "evaluation_data_summary.json",
    }
    for path in saved_paths.values():
        assert path.is_file()
        assert json.loads(path.read_text(encoding="utf-8"))[
            "data_version"
        ] == "fixture-v1"

    restored = load_evaluation_handoff(tmp_path)
    assert restored == handoff


def test_real_movielens_handoff_if_available() -> None:
    """Lock the audited MovieLens evaluation eligibility counts."""
    processed = PROJECT_ROOT / "data" / "processed"
    required_paths = [
        processed / "train.csv",
        processed / "validation.csv",
        processed / "test.csv",
        processed / "id_mappings.json",
    ]
    if not all(path.is_file() for path in required_paths):
        pytest.skip("Processed MovieLens data is not available")

    user_mapping, movie_mapping = load_id_mappings(required_paths[3])
    handoff = build_evaluation_handoff(
        load_processed_ratings(required_paths[0]),
        load_processed_ratings(required_paths[1]),
        load_processed_ratings(required_paths[2]),
        user_mapping,
        movie_mapping,
        data_version="ml100k-temporal-v1",
        positive_threshold=4.0,
        top_k=10,
        negative_sample_size=100,
        random_seed=42,
    )

    assert handoff.summary["users"]["total"] == 943
    assert handoff.summary["users"]["evaluable"] == 836
    assert handoff.summary["users"]["skipped_without_positive"] == 107
    assert handoff.summary["catalog"]["total_movies"] == 1682
    assert handoff.summary["catalog"]["warm_start_movies"] == 1611
    assert handoff.summary["catalog"]["test_cold_start_movies"] == 45
