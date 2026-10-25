"""Build deterministic evaluation inputs from processed interactions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from cinematch.data.mapping import IdMapping
from cinematch.data.schema import MAPPED_RATING_COLUMNS
from cinematch.evaluation.candidates import (
    build_positive_items,
    build_seen_items,
)


HANDOFF_SCHEMA_VERSION = "1.0"
CATALOG_FILENAME = "catalog.json"
SEEN_ITEMS_FILENAME = "seen_items.json"
POSITIVE_ITEMS_FILENAME = "positive_test_items.json"
SUMMARY_FILENAME = "evaluation_data_summary.json"


class EvaluationHandoffError(ValueError):
    """Raised when evaluation inputs violate the shared contract."""


@dataclass(frozen=True)
class EvaluationHandoff:
    """Four JSON documents handed from Data to Evaluation."""

    catalog: dict[str, Any]
    seen_items: dict[str, Any]
    positive_test_items: dict[str, Any]
    summary: dict[str, Any]

    def documents(self) -> dict[str, dict[str, Any]]:
        """Return output filenames mapped to JSON documents."""
        return {
            CATALOG_FILENAME: self.catalog,
            SEEN_ITEMS_FILENAME: self.seen_items,
            POSITIVE_ITEMS_FILENAME: self.positive_test_items,
            SUMMARY_FILENAME: self.summary,
        }


def _validate_partition(
    frame: pd.DataFrame,
    partition_name: str,
    user_mapping: IdMapping,
    movie_mapping: IdMapping,
) -> None:
    """Validate schema, uniqueness and ID/index agreement."""
    actual_columns = tuple(frame.columns)
    if actual_columns != MAPPED_RATING_COLUMNS:
        raise EvaluationHandoffError(
            f"{partition_name} has unexpected columns: "
            f"expected {MAPPED_RATING_COLUMNS}, "
            f"received {actual_columns}"
        )
    if frame.empty:
        raise EvaluationHandoffError(f"{partition_name} is empty")
    if frame.isna().any().any():
        raise EvaluationHandoffError(
            f"{partition_name} contains missing values"
        )
    if frame.duplicated(["user_id", "movie_id"]).any():
        raise EvaluationHandoffError(
            f"{partition_name} contains duplicate user-movie pairs"
        )

    expected_user_indices = user_mapping.encode(frame["user_id"])
    expected_movie_indices = movie_mapping.encode(frame["movie_id"])
    if not expected_user_indices.equals(frame["user_index"]):
        raise EvaluationHandoffError(
            f"{partition_name} has inconsistent user IDs and indices"
        )
    if not expected_movie_indices.equals(frame["movie_index"]):
        raise EvaluationHandoffError(
            f"{partition_name} has inconsistent movie IDs and indices"
        )


def _movie_indices_by_user(
    frame: pd.DataFrame,
) -> dict[int, list[int]]:
    """Group movie indices by user index with deterministic order."""
    return {
        int(user_index): sorted(
            int(value)
            for value in user_rows["movie_index"].tolist()
        )
        for user_index, user_rows in frame.groupby(
            "user_index",
            sort=True,
        )
    }


def _test_rows_by_user(
    test: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    """Group test interactions by user index."""
    return {
        int(user_index): user_rows
        for user_index, user_rows in test.groupby(
            "user_index",
            sort=True,
        )
    }


def _utc_timestamp(generated_at: datetime | None) -> str:
    """Return one validated UTC timestamp for every output document."""
    timestamp = generated_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise EvaluationHandoffError(
            "generated_at must be timezone-aware"
        )
    return (
        timestamp.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def build_evaluation_handoff(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    user_mapping: IdMapping,
    movie_mapping: IdMapping,
    *,
    data_version: str,
    positive_threshold: float,
    top_k: int,
    negative_sample_size: int,
    random_seed: int,
    generated_at: datetime | None = None,
) -> EvaluationHandoff:
    """Build catalog, seen, positive and summary evaluation artifacts."""
    for partition_name, frame in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        _validate_partition(
            frame,
            partition_name,
            user_mapping,
            movie_mapping,
        )

    if top_k <= 0:
        raise EvaluationHandoffError("top_k must be positive")
    if negative_sample_size <= 0:
        raise EvaluationHandoffError(
            "negative_sample_size must be positive"
        )
    if not data_version.strip():
        raise EvaluationHandoffError("data_version cannot be empty")

    all_user_indices = set(range(user_mapping.size))
    observed_user_indices = set(
        pd.concat(
            [
                train["user_index"],
                validation["user_index"],
                test["user_index"],
            ],
            ignore_index=True,
        ).astype(int)
    )
    if observed_user_indices != all_user_indices:
        raise EvaluationHandoffError(
            "Prepared interactions do not cover the complete user mapping"
        )

    train_by_user = _movie_indices_by_user(train)
    validation_by_user = _movie_indices_by_user(validation)
    test_by_user = _test_rows_by_user(test)
    train_movie_indices = {
        int(value) for value in train["movie_index"].unique()
    }

    seen_users: list[dict[str, Any]] = []
    positive_users: list[dict[str, Any]] = []
    evaluable_user_indices: list[int] = []
    skipped_user_indices: list[int] = []
    total_positive_items = 0

    for user_index, user_id in enumerate(user_mapping.external_ids):
        seen = build_seen_items(
            train_movie_indices=train_by_user.get(user_index, []),
            validation_movie_indices=validation_by_user.get(
                user_index,
                [],
            ),
        )
        user_test = test_by_user.get(user_index)
        if user_test is None:
            test_movie_indices: list[int] = []
            test_ratings: list[float] = []
        else:
            test_movie_indices = [
                int(value)
                for value in user_test["movie_index"].tolist()
            ]
            test_ratings = [
                float(value)
                for value in user_test["rating"].tolist()
            ]

        positives = build_positive_items(
            test_movie_indices=test_movie_indices,
            test_ratings=test_ratings,
            positive_threshold=positive_threshold,
        )
        if seen.intersection(positives):
            raise EvaluationHandoffError(
                "Positive test items must not overlap seen items"
            )

        seen_users.append(
            {
                "user_id": int(user_id),
                "user_index": user_index,
                "movie_indices": sorted(seen),
            }
        )
        positive_users.append(
            {
                "user_id": int(user_id),
                "user_index": user_index,
                "movie_indices": sorted(positives),
            }
        )
        total_positive_items += len(positives)
        if positives:
            evaluable_user_indices.append(user_index)
        else:
            skipped_user_indices.append(user_index)

    catalog_movie_indices = set(range(movie_mapping.size))
    validation_movie_indices = {
        int(value) for value in validation["movie_index"].unique()
    }
    test_movie_indices_set = {
        int(value) for value in test["movie_index"].unique()
    }
    validation_cold_start = sorted(
        validation_movie_indices - train_movie_indices
    )
    test_cold_start = sorted(
        test_movie_indices_set - train_movie_indices
    )
    generated_at_utc = _utc_timestamp(generated_at)
    common = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc,
        "data_version": data_version,
    }

    catalog_document = {
        **common,
        "artifact_type": "evaluation_catalog",
        "number_of_movies": movie_mapping.size,
        "movies": [
            {
                "movie_id": int(movie_id),
                "movie_index": movie_index,
                "observed_in_train": movie_index
                in train_movie_indices,
            }
            for movie_index, movie_id in enumerate(
                movie_mapping.external_ids
            )
        ],
    }
    seen_document = {
        **common,
        "artifact_type": "seen_items",
        "policy": "train_union_validation",
        "users": seen_users,
    }
    positive_document = {
        **common,
        "artifact_type": "positive_test_items",
        "positive_rating_threshold": positive_threshold,
        "users": positive_users,
    }
    summary_document = {
        **common,
        "artifact_type": "evaluation_data_summary",
        "protocol": {
            "seen_items": "train_union_validation",
            "positive_rating_threshold": positive_threshold,
            "top_k": top_k,
            "negative_sample_size": negative_sample_size,
            "random_seed": random_seed,
            "id_space": "zero_based_model_indices",
        },
        "split_rows": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "users": {
            "total": user_mapping.size,
            "evaluable": len(evaluable_user_indices),
            "skipped_without_positive": len(skipped_user_indices),
            "evaluable_user_indices": evaluable_user_indices,
            "skipped_user_indices": skipped_user_indices,
        },
        "positives": {
            "total_test_items": total_positive_items,
        },
        "catalog": {
            "total_movies": len(catalog_movie_indices),
            "warm_start_movies": len(train_movie_indices),
            "validation_cold_start_movies": len(
                validation_cold_start
            ),
            "test_cold_start_movies": len(test_cold_start),
            "validation_cold_start_movie_indices": (
                validation_cold_start
            ),
            "test_cold_start_movie_indices": test_cold_start,
            "validation_cold_start_rows": int(
                validation["movie_index"]
                .isin(validation_cold_start)
                .sum()
            ),
            "test_cold_start_rows": int(
                test["movie_index"]
                .isin(test_cold_start)
                .sum()
            ),
        },
        "files": {
            "catalog": CATALOG_FILENAME,
            "seen_items": SEEN_ITEMS_FILENAME,
            "positive_test_items": POSITIVE_ITEMS_FILENAME,
        },
    }

    return EvaluationHandoff(
        catalog=catalog_document,
        seen_items=seen_document,
        positive_test_items=positive_document,
        summary=summary_document,
    )


def save_evaluation_handoff(
    handoff: EvaluationHandoff,
    output_directory: Path,
) -> dict[str, Path]:
    """Save all handoff documents and return their resolved paths."""
    resolved_directory = output_directory.expanduser().resolve()
    resolved_directory.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, Path] = {}

    for filename, document in handoff.documents().items():
        output_path = resolved_directory / filename
        output_path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        saved_paths[filename] = output_path

    return saved_paths


def load_evaluation_handoff(
    input_directory: Path,
) -> EvaluationHandoff:
    """Load and validate a previously generated handoff bundle."""
    resolved_directory = input_directory.expanduser().resolve()
    expected_types = {
        CATALOG_FILENAME: "evaluation_catalog",
        SEEN_ITEMS_FILENAME: "seen_items",
        POSITIVE_ITEMS_FILENAME: "positive_test_items",
        SUMMARY_FILENAME: "evaluation_data_summary",
    }
    documents: dict[str, dict[str, Any]] = {}

    for filename, expected_type in expected_types.items():
        input_path = resolved_directory / filename
        if not input_path.is_file():
            raise FileNotFoundError(
                f"Evaluation handoff file was not found: {input_path}"
            )
        try:
            document = json.loads(input_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise EvaluationHandoffError(
                f"Invalid evaluation handoff JSON: {input_path}"
            ) from error
        if not isinstance(document, dict):
            raise EvaluationHandoffError(
                f"Evaluation handoff must be a JSON object: {input_path}"
            )
        if document.get("schema_version") != HANDOFF_SCHEMA_VERSION:
            raise EvaluationHandoffError(
                f"Unsupported handoff schema version in {filename}"
            )
        if document.get("artifact_type") != expected_type:
            raise EvaluationHandoffError(
                f"Unexpected artifact_type in {filename}"
            )
        documents[filename] = document

    data_versions = {
        document.get("data_version")
        for document in documents.values()
    }
    generation_times = {
        document.get("generated_at_utc")
        for document in documents.values()
    }
    if len(data_versions) != 1 or None in data_versions:
        raise EvaluationHandoffError(
            "Evaluation handoff files use different data versions"
        )
    if len(generation_times) != 1 or None in generation_times:
        raise EvaluationHandoffError(
            "Evaluation handoff files were not generated together"
        )

    catalog = documents[CATALOG_FILENAME]
    seen_items = documents[SEEN_ITEMS_FILENAME]
    positive_items = documents[POSITIVE_ITEMS_FILENAME]
    summary = documents[SUMMARY_FILENAME]
    try:
        total_users = int(summary["users"]["total"])
        catalog_size = int(summary["catalog"]["total_movies"])
        seen_user_count = len(seen_items["users"])
        positive_user_count = len(positive_items["users"])
        catalog_entry_count = len(catalog["movies"])
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationHandoffError(
            "Evaluation handoff has an invalid document structure"
        ) from error

    if seen_user_count != total_users or positive_user_count != total_users:
        raise EvaluationHandoffError(
            "Evaluation handoff does not contain every mapped user"
        )
    if catalog_entry_count != catalog_size:
        raise EvaluationHandoffError(
            "Evaluation catalog size does not match its summary"
        )

    return EvaluationHandoff(
        catalog=catalog,
        seen_items=seen_items,
        positive_test_items=positive_items,
        summary=summary,
    )
