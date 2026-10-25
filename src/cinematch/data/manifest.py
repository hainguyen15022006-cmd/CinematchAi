"""Build a reproducible manifest for prepared CineMatch data."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from cinematch.data.auditing import audit_temporal_splits
from cinematch.data.configuration import DataPipelineConfig
from cinematch.data.schema import (
    GENRE_COLUMNS,
    MAPPED_RATING_COLUMNS,
    PROCESSED_MOVIE_COLUMNS,
)
from cinematch.features.hybrid_features import (
    GENRE_FEATURE_DIM,
    HISTORY_FEATURE_DIM,
    HYBRID_SIDE_FEATURE_DIM,
    TEXT_FEATURE_DIM,
    YEAR_FEATURE_DIM,
)


class DataManifestError(ValueError):
    """Raised when prepared artifacts do not satisfy the data contract."""


CONTENT_HASH_POLICY = "cinematch-content-v1"


def _canonical_json_sha256(payload: Any) -> str:
    """Hash JSON deterministically without changing array order or values."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _content_sha256(path: Path, file_sha256: str) -> str:
    """Ignore only root-level JSON generation time, never data timestamps.

    Non-JSON artifacts retain their exact-byte hash. JSON whitespace and
    object-key ordering are normalized; list ordering remains significant.
    """
    if path.suffix.lower() != ".json":
        return file_sha256
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload.pop("generated_at_utc", None)
        return _canonical_json_sha256(payload)
    except (UnicodeError, ValueError) as error:
        raise DataManifestError(
            f"Cannot fingerprint invalid JSON artifact: {path}"
        ) from error


def _load_csv_with_columns(
    path: Path,
    expected_columns: tuple[str, ...],
    artifact_name: str,
) -> pd.DataFrame:
    """Load a required CSV and enforce its ordered column contract."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{artifact_name} was not found: {path}"
        )

    frame = pd.read_csv(path)
    actual_columns = tuple(frame.columns)
    if actual_columns != expected_columns:
        raise DataManifestError(
            f"{artifact_name} has unexpected columns: "
            f"expected {expected_columns}, received {actual_columns}"
        )
    if frame.empty:
        raise DataManifestError(f"{artifact_name} is empty")
    return frame


def _sha256(path: Path) -> str:
    """Return the SHA-256 checksum of one artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(
            lambda: artifact_file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path, project_root: Path) -> str:
    """Return a repository-relative POSIX path."""
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError as error:
        raise DataManifestError(
            f"Artifact path escapes the project root: {path}"
        ) from error


def _artifact_entry(
    path: Path,
    project_root: Path,
    rows: int | None = None,
) -> dict[str, Any]:
    """Describe a file without embedding its contents in the manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"Data artifact was not found: {path}")

    entry: dict[str, Any] = {
        "path": _relative_path(path, project_root),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    entry["content_sha256"] = _content_sha256(path, entry["sha256"])
    if rows is not None:
        entry["rows"] = rows
    return entry


def _mapping_sizes(path: Path) -> tuple[int, int]:
    """Read and validate the two mapping cardinalities."""
    if not path.is_file():
        raise FileNotFoundError(f"ID mappings were not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        user_ids = payload["users"]["external_ids"]
        movie_ids = payload["movies"]["external_ids"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise DataManifestError(
            f"Invalid ID mapping artifact: {path}"
        ) from error

    if not isinstance(user_ids, list) or not isinstance(movie_ids, list):
        raise DataManifestError(
            "ID mapping external_ids fields must be lists"
        )
    if len(set(user_ids)) != len(user_ids):
        raise DataManifestError("User mapping contains duplicate IDs")
    if len(set(movie_ids)) != len(movie_ids):
        raise DataManifestError("Movie mapping contains duplicate IDs")
    return len(user_ids), len(movie_ids)


def build_data_manifest(
    config: DataPipelineConfig,
    project_root: Path,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Calculate an honest manifest from the generated data artifacts."""
    root = project_root.expanduser().resolve()
    if (
        config.feature_contract_dimensions
        != HYBRID_SIDE_FEATURE_DIM
    ):
        raise DataManifestError(
            "Configured feature dimensions do not match code: "
            f"expected {config.feature_contract_dimensions}, "
            f"received {HYBRID_SIDE_FEATURE_DIM}"
        )

    train = _load_csv_with_columns(
        config.paths.train,
        MAPPED_RATING_COLUMNS,
        "train split",
    )
    validation = _load_csv_with_columns(
        config.paths.validation,
        MAPPED_RATING_COLUMNS,
        "validation split",
    )
    test = _load_csv_with_columns(
        config.paths.test,
        MAPPED_RATING_COLUMNS,
        "test split",
    )
    movies = _load_csv_with_columns(
        config.paths.movies_processed,
        PROCESSED_MOVIE_COLUMNS,
        "processed movie catalog",
    )

    audit = audit_temporal_splits(
        train,
        validation,
        test,
        positive_threshold=config.positive_rating_threshold,
    )
    all_interactions = pd.concat(
        [train, validation, test],
        ignore_index=True,
    )
    actual_counts = {
        "ratings": len(all_interactions),
        "users": int(all_interactions["user_id"].nunique()),
        "movies": len(movies),
    }
    expected_counts = {
        "ratings": config.expected_ratings,
        "users": config.expected_users,
        "movies": config.expected_movies,
    }
    if actual_counts != expected_counts:
        raise DataManifestError(
            "Prepared data counts do not match configuration: "
            f"expected {expected_counts}, received {actual_counts}"
        )

    mapping_users, mapping_movies = _mapping_sizes(
        config.paths.mappings
    )
    if mapping_users != actual_counts["users"]:
        raise DataManifestError(
            "User mapping size does not match prepared interactions"
        )
    if mapping_movies != actual_counts["movies"]:
        raise DataManifestError(
            "Movie mapping size does not match the processed catalog"
        )

    timestamp = generated_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise DataManifestError("generated_at must be timezone-aware")
    timestamp = timestamp.astimezone(timezone.utc)

    downstream_artifacts = {
        "evaluation_catalog": (
            config.paths.evaluation_handoff_dir / "catalog.json"
        ),
        "evaluation_seen_items": (
            config.paths.evaluation_handoff_dir / "seen_items.json"
        ),
        "evaluation_positive_test_items": (
            config.paths.evaluation_handoff_dir
            / "positive_test_items.json"
        ),
        "evaluation_summary": (
            config.paths.evaluation_handoff_dir
            / "evaluation_data_summary.json"
        ),
        "movie_numeric_features": config.paths.movie_numeric_features,
        "user_genre_profiles": config.paths.user_genre_profiles,
        "numeric_feature_preprocessor": (
            config.paths.numeric_feature_preprocessor
        ),
        "user_pseudo_text": config.paths.user_pseudo_text,
        "movie_text": config.paths.movie_text,
        "user_text_vectors": config.paths.user_text_vectors,
        "movie_text_vectors": config.paths.movie_text_vectors,
        "text_feature_preprocessor": (
            config.paths.text_feature_preprocessor
        ),
    }
    missing = sorted(
        name
        for name, path in downstream_artifacts.items()
        if not path.expanduser().resolve().is_file()
    )
    if missing:
        raise DataManifestError(
            "The manifest is created after the whole pipeline; these "
            f"artifacts are missing: {missing}. Run the evaluation and "
            "feature scripts first."
        )

    manifest = {
        "schema_version": "1.1",
        "generated_at_utc": timestamp.isoformat().replace(
            "+00:00",
            "Z",
        ),
        "dataset": {
            "name": config.dataset_name,
            "version": config.data_version,
            "rating_scale": [1.0, 5.0],
        },
        "feature_contract": {
            "version": config.feature_contract_version,
            "ordered_layout": [
                {"name": "movie_genres", "dimensions": GENRE_FEATURE_DIM},
                {"name": "normalized_release_year", "dimensions": YEAR_FEATURE_DIM},
                {"name": "user_genre_history", "dimensions": HISTORY_FEATURE_DIM},
                {
                    "name": "user_movie_text_interaction",
                    "dimensions": TEXT_FEATURE_DIM,
                },
            ],
            "total_dimensions": HYBRID_SIDE_FEATURE_DIM,
            "genre_names": list(GENRE_COLUMNS),
        },
        "entities": {
            **actual_counts,
            "rated_movies": int(
                all_interactions["movie_id"].nunique()
            ),
        },
        "split": {
            "strategy": "per_user_temporal",
            "order_column": "timestamp",
            "tie_breaker_column": "movie_id",
            "allocation_method": "largest_remainder",
            "ratios": {
                "train": config.train_ratio,
                "validation": config.validation_ratio,
                "test": config.test_ratio,
            },
            "rows": {
                "train": audit.train.rows,
                "validation": audit.validation.rows,
                "test": audit.test.rows,
            },
        },
        "quality": {
            "positive_rating_threshold": config.positive_rating_threshold,
            "train_validation_overlap": audit.train_validation_overlap,
            "train_test_overlap": audit.train_test_overlap,
            "validation_test_overlap": audit.validation_test_overlap,
            "train_validation_temporal_violations": (
                audit.train_validation_temporal_violations
            ),
            "validation_test_temporal_violations": (
                audit.validation_test_temporal_violations
            ),
            "validation_cold_start_movies": len(
                audit.validation_cold_start_movies
            ),
            "test_cold_start_movies": len(
                audit.test_cold_start_movies
            ),
            "validation_cold_start_rows": (
                audit.validation_cold_start_rows
            ),
            "test_cold_start_rows": audit.test_cold_start_rows,
        },
        "evaluation_contract": {
            "seen_items": "train_union_validation",
            "positive_rating_threshold": (
                config.positive_rating_threshold
            ),
            "top_k": config.evaluation_top_k,
            "negative_sample_size": config.negative_sample_size,
            "random_seed": config.random_seed,
        },
        "artifacts": {
            "raw_ratings": _artifact_entry(
                config.paths.ratings_raw,
                root,
                rows=actual_counts["ratings"],
            ),
            "raw_movies": _artifact_entry(
                config.paths.movies_raw,
                root,
                rows=actual_counts["movies"],
            ),
            "train": _artifact_entry(
                config.paths.train,
                root,
                rows=len(train),
            ),
            "validation": _artifact_entry(
                config.paths.validation,
                root,
                rows=len(validation),
            ),
            "test": _artifact_entry(
                config.paths.test,
                root,
                rows=len(test),
            ),
            "movies": _artifact_entry(
                config.paths.movies_processed,
                root,
                rows=len(movies),
            ),
            "id_mappings": _artifact_entry(
                config.paths.mappings,
                root,
            ),
            **{
                name: _artifact_entry(path, root)
                for name, path in downstream_artifacts.items()
            },
        },
        "feature_generation": {
            "random_seed": config.random_seed,
            "positive_rating_threshold": (
                config.positive_rating_threshold
            ),
            "pseudo_text": {
                "language": config.pseudo_text_language,
                "maximum_genres": config.pseudo_text_maximum_genres,
                "minimum_genre_observations": (
                    config.pseudo_text_minimum_genre_observations
                ),
            },
        },
    }
    # Exclude file sizes, paths, byte checksums and generation timestamps:
    # they can vary between equivalent reruns. Include the actual data,
    # feature recipe and evaluation contract, not just the version labels.
    reproducible_payload = {
        key: manifest[key]
        for key in (
            "schema_version", "dataset", "feature_contract", "entities",
            "split", "quality", "evaluation_contract", "feature_generation",
        )
    }
    reproducible_payload["artifacts"] = {
        name: entry["content_sha256"]
        for name, entry in manifest["artifacts"].items()
    }
    reproducible_payload["canonicalization"] = CONTENT_HASH_POLICY
    manifest["reproducibility"] = {
        "algorithm": "sha256",
        "canonicalization": CONTENT_HASH_POLICY,
        "content_sha256": _canonical_json_sha256(reproducible_payload),
    }
    return manifest


def save_data_manifest(
    manifest: dict[str, Any],
    output_path: Path,
) -> Path:
    """Write a manifest as readable UTF-8 JSON."""
    resolved_output = output_path.expanduser().resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return resolved_output
