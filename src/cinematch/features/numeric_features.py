"""Leakage-safe numeric side features for CineMatch Hybrid NCF."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from cinematch.data.mapping import IdMapping
from cinematch.data.schema import (
    GENRE_COLUMNS,
    MAPPED_RATING_COLUMNS,
    PROCESSED_MOVIE_COLUMNS,
)


NUMERIC_FEATURE_SCHEMA_VERSION = "1.0"
NORMALIZED_YEAR_COLUMN = "normalized_release_year"
HISTORY_COLUMNS = tuple(
    f"history_{genre}" for genre in GENRE_COLUMNS
)
MOVIE_NUMERIC_COLUMNS = (
    "movie_id",
    "movie_index",
    *GENRE_COLUMNS,
    NORMALIZED_YEAR_COLUMN,
)
USER_GENRE_PROFILE_COLUMNS = (
    "user_id",
    "user_index",
    *HISTORY_COLUMNS,
)
NUMERIC_FEATURE_COLUMNS = (
    *GENRE_COLUMNS,
    NORMALIZED_YEAR_COLUMN,
    *HISTORY_COLUMNS,
)
NUMERIC_FEATURE_DIM = len(NUMERIC_FEATURE_COLUMNS)


class NumericFeatureError(ValueError):
    """Raised when numeric feature inputs or artifacts are invalid."""


@dataclass(frozen=True)
class ReleaseYearScaler:
    """Train-only statistics used to standardize release years."""

    median: float
    mean: float
    standard_deviation: float
    fit_movie_count: int
    fit_non_missing_count: int

    def __post_init__(self) -> None:
        values = (
            self.median,
            self.mean,
            self.standard_deviation,
        )
        if not all(math.isfinite(value) for value in values):
            raise NumericFeatureError(
                "Release-year statistics must be finite"
            )
        if self.standard_deviation <= 0.0:
            raise NumericFeatureError(
                "Release-year standard deviation must be positive"
            )
        if self.fit_movie_count <= 0:
            raise NumericFeatureError(
                "Release-year scaler requires training movies"
            )
        if not 0 < self.fit_non_missing_count <= self.fit_movie_count:
            raise NumericFeatureError(
                "Invalid non-missing release-year count"
            )

    def as_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-compatible scaler contract."""
        return {
            "fit_partition": "train",
            "missing_strategy": "train_median",
            "scaling": "z_score_population",
            **asdict(self),
        }

    @classmethod
    def from_dict(cls, payload: object) -> "ReleaseYearScaler":
        """Load scaler statistics from a JSON object."""
        if not isinstance(payload, dict):
            raise NumericFeatureError(
                "release_year must be a JSON object"
            )
        if payload.get("fit_partition") != "train":
            raise NumericFeatureError(
                "Release-year scaler must be fit on train"
            )
        if payload.get("missing_strategy") != "train_median":
            raise NumericFeatureError(
                "Unsupported release-year missing strategy"
            )
        if payload.get("scaling") != "z_score_population":
            raise NumericFeatureError(
                "Unsupported release-year scaling strategy"
            )
        try:
            return cls(
                median=float(payload["median"]),
                mean=float(payload["mean"]),
                standard_deviation=float(
                    payload["standard_deviation"]
                ),
                fit_movie_count=int(payload["fit_movie_count"]),
                fit_non_missing_count=int(
                    payload["fit_non_missing_count"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise NumericFeatureError(
                "Invalid release-year scaler artifact"
            ) from error


@dataclass(frozen=True)
class NumericFeatureArtifacts:
    """Movie features, user histories and their preprocessing contract."""

    movie_features: pd.DataFrame
    user_profiles: pd.DataFrame
    preprocessor: dict[str, Any]


def _validate_frame_columns(
    frame: pd.DataFrame,
    expected_columns: tuple[str, ...],
    frame_name: str,
) -> None:
    """Require an exact ordered DataFrame schema and non-empty data."""
    actual_columns = tuple(frame.columns)
    if actual_columns != expected_columns:
        raise NumericFeatureError(
            f"{frame_name} has unexpected columns: "
            f"expected {expected_columns}, received {actual_columns}"
        )
    if frame.empty:
        raise NumericFeatureError(f"{frame_name} is empty")


def _validate_train_and_movies(
    train: pd.DataFrame,
    movies: pd.DataFrame,
    user_mapping: IdMapping,
    movie_mapping: IdMapping,
) -> None:
    """Validate source schemas, uniqueness and ID/index consistency."""
    _validate_frame_columns(
        train,
        MAPPED_RATING_COLUMNS,
        "train interactions",
    )
    _validate_frame_columns(
        movies,
        PROCESSED_MOVIE_COLUMNS,
        "processed movie catalog",
    )
    if train.isna().any().any():
        raise NumericFeatureError(
            "train interactions contain missing values"
        )
    if train.duplicated(["user_id", "movie_id"]).any():
        raise NumericFeatureError(
            "train interactions contain duplicate user-movie pairs"
        )
    if movies["movie_id"].duplicated().any():
        raise NumericFeatureError(
            "processed movie catalog contains duplicate movie IDs"
        )
    if len(movies) != movie_mapping.size:
        raise NumericFeatureError(
            "movie catalog size does not match movie mapping"
        )

    expected_user_indices = user_mapping.encode(train["user_id"])
    expected_movie_indices = movie_mapping.encode(train["movie_id"])
    if not expected_user_indices.equals(train["user_index"]):
        raise NumericFeatureError(
            "train interactions have inconsistent user IDs and indices"
        )
    if not expected_movie_indices.equals(train["movie_index"]):
        raise NumericFeatureError(
            "train interactions have inconsistent movie IDs and indices"
        )

    expected_catalog_indices = movie_mapping.encode(movies["movie_id"])
    if set(expected_catalog_indices) != set(range(movie_mapping.size)):
        raise NumericFeatureError(
            "processed movie catalog does not cover the movie mapping"
        )

    genre_values = movies.loc[:, list(GENRE_COLUMNS)]
    if not genre_values.isin([0, 1]).all().all():
        raise NumericFeatureError(
            "movie genre columns must contain only zero or one"
        )


def fit_release_year_scaler(
    train: pd.DataFrame,
    movies: pd.DataFrame,
) -> ReleaseYearScaler:
    """Fit release-year imputation and scaling on train movies only."""
    train_movie_ids = train["movie_id"].drop_duplicates()
    training_movies = movies.loc[
        movies["movie_id"].isin(train_movie_ids),
        ["movie_id", "release_year"],
    ].copy()
    if len(training_movies) != train_movie_ids.nunique():
        raise NumericFeatureError(
            "Some training movies are missing from the catalog"
        )

    valid_years = pd.to_numeric(
        training_movies["release_year"],
        errors="coerce",
    ).dropna()
    if valid_years.empty:
        raise NumericFeatureError(
            "Training movies do not contain a valid release year"
        )
    if (valid_years <= 0).any():
        raise NumericFeatureError(
            "Release years must be positive"
        )

    median = float(valid_years.median())
    imputed_years = pd.to_numeric(
        training_movies["release_year"],
        errors="coerce",
    ).fillna(median)
    mean = float(imputed_years.mean())
    standard_deviation = float(imputed_years.std(ddof=0))

    return ReleaseYearScaler(
        median=median,
        mean=mean,
        standard_deviation=standard_deviation,
        fit_movie_count=len(training_movies),
        fit_non_missing_count=len(valid_years),
    )


def transform_movie_numeric_features(
    movies: pd.DataFrame,
    movie_mapping: IdMapping,
    scaler: ReleaseYearScaler,
) -> pd.DataFrame:
    """Create 19 genre values and one scaled year for every movie."""
    movie_indices = movie_mapping.encode(movies["movie_id"])
    release_years = pd.to_numeric(
        movies["release_year"],
        errors="coerce",
    ).fillna(scaler.median)
    normalized_years = (
        (release_years - scaler.mean)
        / scaler.standard_deviation
    )

    transformed = pd.DataFrame(
        {
            "movie_id": movies["movie_id"].astype("int64"),
            "movie_index": movie_indices,
        }
    )
    for genre in GENRE_COLUMNS:
        transformed[genre] = movies[genre].astype("float32")
    transformed[NORMALIZED_YEAR_COLUMN] = normalized_years.astype(
        "float32"
    )
    transformed = transformed.loc[
        :, list(MOVIE_NUMERIC_COLUMNS)
    ].sort_values("movie_index", ignore_index=True)

    numeric_values = transformed.loc[
        :,
        [*GENRE_COLUMNS, NORMALIZED_YEAR_COLUMN],
    ].to_numpy(dtype="float32")
    if not torch.isfinite(torch.from_numpy(numeric_values)).all():
        raise NumericFeatureError(
            "Movie numeric features contain non-finite values"
        )
    return transformed


def build_user_genre_profiles(
    train: pd.DataFrame,
    movies: pd.DataFrame,
    user_mapping: IdMapping,
) -> pd.DataFrame:
    """Aggregate train-only centered ratings into 19 genre scores."""
    joined = train.loc[
        :, ["user_index", "movie_id", "rating"]
    ].merge(
        movies.loc[:, ["movie_id", *GENRE_COLUMNS]],
        on="movie_id",
        how="left",
        validate="many_to_one",
    )
    if joined.loc[:, list(GENRE_COLUMNS)].isna().any().any():
        raise NumericFeatureError(
            "Some training movies are missing genre metadata"
        )

    centered_ratings = (joined["rating"].astype(float) - 3.0) / 2.0
    genre_membership = joined.loc[
        :, list(GENRE_COLUMNS)
    ].astype(float)
    contributions = genre_membership.mul(
        centered_ratings,
        axis=0,
    )
    contributions["user_index"] = joined["user_index"].to_numpy()
    membership_counts = genre_membership.copy()
    membership_counts["user_index"] = joined["user_index"].to_numpy()

    contribution_sums = contributions.groupby(
        "user_index",
        sort=True,
    )[list(GENRE_COLUMNS)].sum()
    count_sums = membership_counts.groupby(
        "user_index",
        sort=True,
    )[list(GENRE_COLUMNS)].sum()
    profile_values = contribution_sums.div(
        count_sums.where(count_sums > 0)
    ).fillna(0.0)
    profile_values = profile_values.reindex(
        range(user_mapping.size),
        fill_value=0.0,
    )

    profiles = pd.DataFrame(
        {
            "user_id": list(user_mapping.external_ids),
            "user_index": list(range(user_mapping.size)),
        }
    )
    for genre, history_column in zip(
        GENRE_COLUMNS,
        HISTORY_COLUMNS,
        strict=True,
    ):
        profiles[history_column] = profile_values[genre].to_numpy(
            dtype="float32"
        )
    profiles = profiles.loc[:, list(USER_GENRE_PROFILE_COLUMNS)]

    history_values = profiles.loc[
        :, list(HISTORY_COLUMNS)
    ].to_numpy(dtype="float32")
    history_tensor = torch.from_numpy(history_values)
    if not torch.isfinite(history_tensor).all():
        raise NumericFeatureError(
            "User genre profiles contain non-finite values"
        )
    if (history_tensor < -1.0).any() or (history_tensor > 1.0).any():
        raise NumericFeatureError(
            "User genre profiles must stay within [-1, 1]"
        )
    return profiles


def build_interaction_numeric_features(
    interactions: pd.DataFrame,
    movie_features: pd.DataFrame,
    user_profiles: pd.DataFrame,
) -> torch.Tensor:
    """Join reusable artifacts into a 39-column interaction tensor."""
    _validate_frame_columns(
        interactions,
        MAPPED_RATING_COLUMNS,
        "interactions",
    )
    _validate_frame_columns(
        movie_features,
        MOVIE_NUMERIC_COLUMNS,
        "movie numeric features",
    )
    _validate_frame_columns(
        user_profiles,
        USER_GENRE_PROFILE_COLUMNS,
        "user genre profiles",
    )
    joined = interactions.loc[
        :, ["user_id", "movie_id", "user_index", "movie_index"]
    ].merge(
        movie_features,
        on=["movie_id", "movie_index"],
        how="left",
        validate="many_to_one",
    ).merge(
        user_profiles,
        on=["user_id", "user_index"],
        how="left",
        validate="many_to_one",
    )
    if joined.loc[:, list(NUMERIC_FEATURE_COLUMNS)].isna().any().any():
        raise NumericFeatureError(
            "Interactions could not be joined to every numeric feature"
        )
    values = joined.loc[
        :, list(NUMERIC_FEATURE_COLUMNS)
    ].to_numpy(dtype="float32")
    tensor = torch.from_numpy(values.copy())
    if tensor.shape[1] != NUMERIC_FEATURE_DIM:
        raise NumericFeatureError(
            f"Expected {NUMERIC_FEATURE_DIM} numeric features"
        )
    if not torch.isfinite(tensor).all():
        raise NumericFeatureError(
            "Interaction numeric features contain non-finite values"
        )
    return tensor


def build_numeric_feature_artifacts(
    train: pd.DataFrame,
    movies: pd.DataFrame,
    user_mapping: IdMapping,
    movie_mapping: IdMapping,
    *,
    data_version: str,
    feature_contract_version: str,
    generated_at: datetime | None = None,
) -> NumericFeatureArtifacts:
    """Fit train-only preprocessing and build reusable feature tables."""
    _validate_train_and_movies(
        train,
        movies,
        user_mapping,
        movie_mapping,
    )
    scaler = fit_release_year_scaler(train, movies)
    movie_features = transform_movie_numeric_features(
        movies,
        movie_mapping,
        scaler,
    )
    user_profiles = build_user_genre_profiles(
        train,
        movies,
        user_mapping,
    )

    timestamp = generated_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise NumericFeatureError(
            "generated_at must be timezone-aware"
        )
    generated_at_utc = (
        timestamp.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    preprocessor = {
        "schema_version": NUMERIC_FEATURE_SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc,
        "data_version": data_version,
        "feature_contract_version": feature_contract_version,
        "fit_partition": "train",
        "numeric_feature_dimensions": NUMERIC_FEATURE_DIM,
        "ordered_feature_columns": list(NUMERIC_FEATURE_COLUMNS),
        "genre_order": list(GENRE_COLUMNS),
        "release_year": scaler.as_dict(),
        "history_profile": {
            "fit_partition": "train",
            "rating_transform": "(rating - 3.0) / 2.0",
            "range": [-1.0, 1.0],
            "unobserved_genre_value": 0.0,
            "aggregation": "mean_per_user_genre",
        },
        "rows": {
            "movie_numeric_features": len(movie_features),
            "user_genre_profiles": len(user_profiles),
        },
    }
    return NumericFeatureArtifacts(
        movie_features=movie_features,
        user_profiles=user_profiles,
        preprocessor=preprocessor,
    )


def save_numeric_feature_artifacts(
    artifacts: NumericFeatureArtifacts,
    movie_features_path: Path,
    user_profiles_path: Path,
    preprocessor_path: Path,
) -> tuple[Path, Path, Path]:
    """Save the three numeric-feature artifacts."""
    resolved_movie_path = movie_features_path.expanduser().resolve()
    resolved_user_path = user_profiles_path.expanduser().resolve()
    resolved_preprocessor_path = preprocessor_path.expanduser().resolve()
    for path in (
        resolved_movie_path,
        resolved_user_path,
        resolved_preprocessor_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    artifacts.movie_features.to_csv(resolved_movie_path, index=False)
    artifacts.user_profiles.to_csv(resolved_user_path, index=False)
    resolved_preprocessor_path.write_text(
        json.dumps(
            artifacts.preprocessor,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return (
        resolved_movie_path,
        resolved_user_path,
        resolved_preprocessor_path,
    )


def load_numeric_feature_artifacts(
    movie_features_path: Path,
    user_profiles_path: Path,
    preprocessor_path: Path,
) -> NumericFeatureArtifacts:
    """Load and validate numeric artifacts for training or serving."""
    for path in (
        movie_features_path,
        user_profiles_path,
        preprocessor_path,
    ):
        if not path.expanduser().resolve().is_file():
            raise FileNotFoundError(
                f"Numeric feature artifact was not found: {path}"
            )

    movie_features = pd.read_csv(movie_features_path)
    user_profiles = pd.read_csv(user_profiles_path)
    try:
        preprocessor = json.loads(
            preprocessor_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise NumericFeatureError(
            "Invalid numeric feature preprocessor JSON"
        ) from error
    if not isinstance(preprocessor, dict):
        raise NumericFeatureError(
            "Numeric feature preprocessor must be a JSON object"
        )

    _validate_frame_columns(
        movie_features,
        MOVIE_NUMERIC_COLUMNS,
        "movie numeric features",
    )
    _validate_frame_columns(
        user_profiles,
        USER_GENRE_PROFILE_COLUMNS,
        "user genre profiles",
    )
    if preprocessor.get("schema_version") != NUMERIC_FEATURE_SCHEMA_VERSION:
        raise NumericFeatureError(
            "Unsupported numeric feature schema version"
        )
    if preprocessor.get("fit_partition") != "train":
        raise NumericFeatureError(
            "Numeric feature preprocessing must be fit on train"
        )
    if preprocessor.get("numeric_feature_dimensions") != NUMERIC_FEATURE_DIM:
        raise NumericFeatureError(
            "Numeric feature dimension does not match code"
        )
    if preprocessor.get("ordered_feature_columns") != list(
        NUMERIC_FEATURE_COLUMNS
    ):
        raise NumericFeatureError(
            "Numeric feature order does not match code"
        )
    ReleaseYearScaler.from_dict(preprocessor.get("release_year"))
    return NumericFeatureArtifacts(
        movie_features=movie_features,
        user_profiles=user_profiles,
        preprocessor=preprocessor,
    )
