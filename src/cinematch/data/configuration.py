"""Validated configuration for the CineMatch data pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when ``cinematch.yaml`` is missing or invalid."""


@dataclass(frozen=True)
class DataPaths:
    """Resolved input, output and report paths."""

    ratings_raw: Path
    movies_raw: Path
    train: Path
    validation: Path
    test: Path
    movies_processed: Path
    mappings: Path
    split_audit: Path
    data_manifest: Path
    evaluation_handoff_dir: Path
    movie_numeric_features: Path
    user_genre_profiles: Path
    numeric_feature_preprocessor: Path
    user_pseudo_text: Path
    movie_text: Path
    user_text_vectors: Path
    movie_text_vectors: Path
    text_feature_preprocessor: Path


@dataclass(frozen=True)
class DataPipelineConfig:
    """Typed values required by data preparation and auditing."""

    random_seed: int
    dataset_name: str
    data_version: str
    feature_contract_version: str
    feature_contract_dimensions: int
    expected_ratings: int
    expected_users: int
    expected_movies: int
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    minimum_interactions_per_user: int
    positive_rating_threshold: float
    evaluation_top_k: int
    negative_sample_size: int
    pseudo_text_language: str
    pseudo_text_maximum_genres: int
    paths: DataPaths


def _require_mapping(
    mapping: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """Return a required nested mapping with a clear error."""
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ConfigurationError(
            f"Configuration key '{key}' must be a mapping"
        )
    return value


def _require_integer(
    mapping: dict[str, Any],
    key: str,
) -> int:
    """Return a required non-boolean integer."""
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(
            f"Configuration key '{key}' must be an integer"
        )
    return value


def _require_string(
    mapping: dict[str, Any],
    key: str,
) -> str:
    """Return a required non-empty string."""
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(
            f"Configuration key '{key}' must be a non-empty string"
        )
    return value.strip()


def _require_number(
    mapping: dict[str, Any],
    key: str,
) -> float:
    """Return a required numeric value as float."""
    value = mapping.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ConfigurationError(
            f"Configuration key '{key}' must be numeric"
        )
    return float(value)


def _resolve_project_path(
    project_root: Path,
    value: object,
    key: str,
) -> Path:
    """Resolve and constrain a configured path to the repository."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(
            f"Configuration key '{key}' must be a path string"
        )

    candidate = (project_root / value).resolve()
    try:
        candidate.relative_to(project_root)
    except ValueError as error:
        raise ConfigurationError(
            f"Configuration path '{key}' escapes the project root"
        ) from error

    return candidate


def load_data_config(
    path: Path,
    project_root: Path | None = None,
) -> DataPipelineConfig:
    """Load and validate the shared YAML data contract."""
    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"CineMatch configuration was not found: {resolved_path}"
        )

    try:
        with resolved_path.open(
            "r",
            encoding="utf-8",
        ) as config_file:
            payload = yaml.safe_load(config_file)
    except yaml.YAMLError as error:
        raise ConfigurationError(
            f"Invalid YAML configuration: {resolved_path}"
        ) from error

    if not isinstance(payload, dict):
        raise ConfigurationError(
            "Configuration root must be a mapping"
        )

    root = (
        project_root.expanduser().resolve()
        if project_root is not None
        else resolved_path.parent.parent.resolve()
    )

    project = _require_mapping(payload, "project")
    data = _require_mapping(payload, "data")
    evaluation = _require_mapping(payload, "evaluation")
    expected = _require_mapping(data, "expected")
    feature_contract = _require_mapping(
        data,
        "feature_contract",
    )
    raw = _require_mapping(data, "raw")
    processed = _require_mapping(data, "processed")
    split = _require_mapping(data, "temporal_split")
    reports = _require_mapping(data, "reports")
    features = _require_mapping(data, "features")
    pseudo_text = _require_mapping(features, "pseudo_text")

    expected_ratings = _require_integer(expected, "ratings")
    expected_users = _require_integer(expected, "users")
    expected_movies = _require_integer(expected, "movies")
    minimum_interactions = _require_integer(
        split,
        "minimum_interactions_per_user",
    )

    if min(
        expected_ratings,
        expected_users,
        expected_movies,
    ) <= 0:
        raise ConfigurationError(
            "Expected dataset counts must be positive"
        )

    if minimum_interactions < 3:
        raise ConfigurationError(
            "minimum_interactions_per_user must be at least 3"
        )

    train_ratio = _require_number(split, "train_ratio")
    validation_ratio = _require_number(
        split,
        "validation_ratio",
    )
    test_ratio = _require_number(split, "test_ratio")

    if any(
        ratio <= 0.0
        for ratio in (
            train_ratio,
            validation_ratio,
            test_ratio,
        )
    ):
        raise ConfigurationError(
            "Temporal split ratios must be positive"
        )

    if abs(
        train_ratio + validation_ratio + test_ratio - 1.0
    ) > 1e-9:
        raise ConfigurationError(
            "Temporal split ratios must sum to 1.0"
        )

    supported_contract = {
        "strategy": "per_user",
        "order_column": "timestamp",
        "tie_breaker_column": "movie_id",
        "allocation_method": "largest_remainder",
    }
    for key, expected_value in supported_contract.items():
        if split.get(key) != expected_value:
            raise ConfigurationError(
                f"Configuration key '{key}' must be "
                f"'{expected_value}'"
            )

    positive_threshold = _require_number(
        evaluation,
        "positive_rating_threshold",
    )
    if not 1.0 <= positive_threshold <= 5.0:
        raise ConfigurationError(
            "positive_rating_threshold must be within [1, 5]"
        )

    evaluation_top_k = _require_integer(evaluation, "top_k")
    negative_sample_size = _require_integer(
        evaluation,
        "negative_sample_size",
    )
    if evaluation_top_k <= 0:
        raise ConfigurationError("top_k must be positive")
    if negative_sample_size <= 0:
        raise ConfigurationError(
            "negative_sample_size must be positive"
        )

    pseudo_text_language = _require_string(
        pseudo_text,
        "language",
    )
    if pseudo_text_language != "en":
        raise ConfigurationError(
            "data.features.pseudo_text.language must be 'en'"
        )
    pseudo_text_maximum_genres = _require_integer(
        pseudo_text,
        "maximum_genres",
    )
    if not 1 <= pseudo_text_maximum_genres <= 19:
        raise ConfigurationError(
            "pseudo-text maximum_genres must be within [1, 19]"
        )

    paths = DataPaths(
        ratings_raw=_resolve_project_path(
            root,
            raw.get("ratings_path"),
            "data.raw.ratings_path",
        ),
        movies_raw=_resolve_project_path(
            root,
            raw.get("movies_path"),
            "data.raw.movies_path",
        ),
        train=_resolve_project_path(
            root,
            processed.get("train_path"),
            "data.processed.train_path",
        ),
        validation=_resolve_project_path(
            root,
            processed.get("validation_path"),
            "data.processed.validation_path",
        ),
        test=_resolve_project_path(
            root,
            processed.get("test_path"),
            "data.processed.test_path",
        ),
        movies_processed=_resolve_project_path(
            root,
            processed.get("movies_path"),
            "data.processed.movies_path",
        ),
        mappings=_resolve_project_path(
            root,
            processed.get("mappings_path"),
            "data.processed.mappings_path",
        ),
        split_audit=_resolve_project_path(
            root,
            reports.get("split_audit_path"),
            "data.reports.split_audit_path",
        ),
        data_manifest=_resolve_project_path(
            root,
            reports.get("data_manifest_path"),
            "data.reports.data_manifest_path",
        ),
        evaluation_handoff_dir=_resolve_project_path(
            root,
            reports.get("evaluation_handoff_dir"),
            "data.reports.evaluation_handoff_dir",
        ),
        movie_numeric_features=_resolve_project_path(
            root,
            features.get("movie_numeric_path"),
            "data.features.movie_numeric_path",
        ),
        user_genre_profiles=_resolve_project_path(
            root,
            features.get("user_genre_profiles_path"),
            "data.features.user_genre_profiles_path",
        ),
        numeric_feature_preprocessor=_resolve_project_path(
            root,
            features.get("numeric_preprocessor_path"),
            "data.features.numeric_preprocessor_path",
        ),
        user_pseudo_text=_resolve_project_path(
            root,
            features.get("user_pseudo_text_path"),
            "data.features.user_pseudo_text_path",
        ),
        movie_text=_resolve_project_path(
            root,
            features.get("movie_text_path"),
            "data.features.movie_text_path",
        ),
        user_text_vectors=_resolve_project_path(
            root,
            features.get("user_text_vectors_path"),
            "data.features.user_text_vectors_path",
        ),
        movie_text_vectors=_resolve_project_path(
            root,
            features.get("movie_text_vectors_path"),
            "data.features.movie_text_vectors_path",
        ),
        text_feature_preprocessor=_resolve_project_path(
            root,
            features.get("text_preprocessor_path"),
            "data.features.text_preprocessor_path",
        ),
    )

    return DataPipelineConfig(
        random_seed=_require_integer(project, "random_seed"),
        dataset_name=_require_string(data, "dataset_name"),
        data_version=_require_string(data, "version"),
        feature_contract_version=_require_string(
            feature_contract,
            "version",
        ),
        feature_contract_dimensions=_require_integer(
            feature_contract,
            "total_dimensions",
        ),
        expected_ratings=expected_ratings,
        expected_users=expected_users,
        expected_movies=expected_movies,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        minimum_interactions_per_user=minimum_interactions,
        positive_rating_threshold=positive_threshold,
        evaluation_top_k=evaluation_top_k,
        negative_sample_size=negative_sample_size,
        pseudo_text_language=pseudo_text_language,
        pseudo_text_maximum_genres=pseudo_text_maximum_genres,
        paths=paths,
    )
