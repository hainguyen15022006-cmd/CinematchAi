"""Deterministic train-only text features for CineMatch Hybrid NCF."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from cinematch.data.mapping import IdMapping
from cinematch.data.schema import (
    GENRE_COLUMNS,
    MAPPED_RATING_COLUMNS,
    PROCESSED_MOVIE_COLUMNS,
)
from cinematch.features.hybrid_features import TEXT_FEATURE_DIM
from cinematch.features.text_encoder import (
    PreferenceTextEncoder,
    TextEncoderConfig,
)


TEXT_FEATURE_SCHEMA_VERSION = "1.0"
TEXT_FUSION = "hadamard_product"
USER_PSEUDO_TEXT_COLUMNS = (
    "user_id",
    "user_index",
    "preferred_genres",
    "used_fallback",
    "pseudo_text",
)
MOVIE_TEXT_COLUMNS = (
    "movie_id",
    "movie_index",
    "genres",
    "movie_text",
)
PSEUDO_TEXT_TEMPLATES = (
    "I enjoy {genres} movies.",
    "My preferred movie genres are {genres}.",
    "I usually choose films in {genres}.",
)


class TextFeatureError(ValueError):
    """Raised when pseudo-text inputs or artifacts are invalid."""


@dataclass(frozen=True)
class TextFeatureArtifacts:
    """Source text, encoded matrices and their preprocessing contract."""

    user_texts: pd.DataFrame
    movie_texts: pd.DataFrame
    user_vectors: np.ndarray
    movie_vectors: np.ndarray
    preprocessor: dict[str, Any]


def _require_exact_columns(
    frame: pd.DataFrame,
    expected: tuple[str, ...],
    name: str,
) -> None:
    """Validate an ordered, non-empty DataFrame schema."""
    actual = tuple(frame.columns)
    if actual != expected:
        raise TextFeatureError(
            f"{name} has unexpected columns: "
            f"expected {expected}, received {actual}"
        )
    if frame.empty:
        raise TextFeatureError(f"{name} is empty")


def _validate_sources(
    train: pd.DataFrame,
    movies: pd.DataFrame,
    user_mapping: IdMapping,
    movie_mapping: IdMapping,
) -> None:
    """Validate training data, catalog and mapping consistency."""
    _require_exact_columns(
        train,
        MAPPED_RATING_COLUMNS,
        "train interactions",
    )
    _require_exact_columns(
        movies,
        PROCESSED_MOVIE_COLUMNS,
        "processed movie catalog",
    )
    if train.isna().any().any():
        raise TextFeatureError("train interactions contain missing values")
    if movies["movie_id"].duplicated().any():
        raise TextFeatureError("movie catalog contains duplicate movie IDs")
    if len(movies) != movie_mapping.size:
        raise TextFeatureError("movie catalog does not match movie mapping")

    expected_user_indices = user_mapping.encode(train["user_id"])
    expected_movie_indices = movie_mapping.encode(train["movie_id"])
    if not expected_user_indices.equals(train["user_index"]):
        raise TextFeatureError("train user IDs and indices are inconsistent")
    if not expected_movie_indices.equals(train["movie_index"]):
        raise TextFeatureError("train movie IDs and indices are inconsistent")
    if train["user_index"].nunique() != user_mapping.size:
        raise TextFeatureError("every mapped user must occur in train")

    catalog_indices = movie_mapping.encode(movies["movie_id"])
    if set(catalog_indices) != set(range(movie_mapping.size)):
        raise TextFeatureError("movie catalog does not cover movie mapping")
    if movies["title"].isna().any() or (
        movies["title"].str.strip() == ""
    ).any():
        raise TextFeatureError("movie titles must not be empty")
    genre_values = movies.loc[:, list(GENRE_COLUMNS)]
    if not genre_values.isin([0, 1]).all().all():
        raise TextFeatureError("genre columns must contain zero or one")


def _format_genres(genres: list[str]) -> str:
    """Format one to three genre names as readable English."""
    if not genres:
        raise TextFeatureError("at least one genre is required")
    if len(genres) == 1:
        return genres[0]
    if len(genres) == 2:
        return f"{genres[0]} and {genres[1]}"
    return f"{', '.join(genres[:-1])} and {genres[-1]}"


def _template_index(user_id: int, seed: int) -> int:
    """Choose a template reproducibly without global random state."""
    digest = hashlib.sha256(f"{seed}:{user_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % len(
        PSEUDO_TEXT_TEMPLATES
    )


def build_user_pseudo_texts(
    train: pd.DataFrame,
    movies: pd.DataFrame,
    user_mapping: IdMapping,
    *,
    positive_rating_threshold: float,
    maximum_genres: int,
    seed: int,
    minimum_genre_observations: int = 1,
) -> pd.DataFrame:
    """Generate English preference sentences from train ratings only.

    A genre is preferred only when its train mean rating reaches the
    positive threshold AND the user rated at least
    ``minimum_genre_observations`` movies of that genre. The count
    condition avoids the small-sample bias where one 5-star rating of a
    rare genre (for example Film-Noir) wins over well-observed genres.
    """
    if not 1.0 <= positive_rating_threshold <= 5.0:
        raise TextFeatureError(
            "positive_rating_threshold must be within [1, 5]"
        )
    if not 1 <= maximum_genres <= len(GENRE_COLUMNS):
        raise TextFeatureError(
            "maximum_genres must be within the genre count"
        )
    if minimum_genre_observations < 1:
        raise TextFeatureError(
            "minimum_genre_observations must be at least 1"
        )

    joined = train.loc[
        :, ["user_index", "movie_id", "rating"]
    ].merge(
        movies.loc[:, ["movie_id", *GENRE_COLUMNS]],
        on="movie_id",
        how="left",
        validate="many_to_one",
    )
    genre_frame = joined.loc[:, list(GENRE_COLUMNS)]
    if genre_frame.isna().any().any():
        raise TextFeatureError(
            "some training movies are missing genre metadata"
        )

    membership = genre_frame.astype("float64")
    weighted = membership.mul(joined["rating"].astype(float), axis=0)
    membership["user_index"] = joined["user_index"].to_numpy()
    weighted["user_index"] = joined["user_index"].to_numpy()
    counts = membership.groupby("user_index", sort=True)[
        list(GENRE_COLUMNS)
    ].sum()
    sums = weighted.groupby("user_index", sort=True)[
        list(GENRE_COLUMNS)
    ].sum()
    counts = counts.reindex(range(user_mapping.size), fill_value=0.0)
    means = sums.reindex(range(user_mapping.size), fill_value=0.0).div(
        counts.where(counts > 0)
    )

    rows: list[dict[str, object]] = []
    for user_index, user_id in enumerate(user_mapping.external_ids):
        candidates: list[tuple[str, float, float]] = []
        for genre in GENRE_COLUMNS:
            if genre == "unknown":
                continue
            count = float(counts.loc[user_index, genre])
            mean = float(means.loc[user_index, genre])
            if count > 0 and np.isfinite(mean):
                candidates.append((genre, mean, count))
        if not candidates:
            raise TextFeatureError(
                f"user_index {user_index} has no observed genres"
            )

        ranked = sorted(
            candidates,
            key=lambda value: (
                -value[1],
                -value[2],
                GENRE_COLUMNS.index(value[0]),
            ),
        )
        preferred = [
            entry
            for entry in ranked
            if entry[1] >= positive_rating_threshold
            and entry[2] >= minimum_genre_observations
        ]
        used_fallback = not preferred
        selected = (preferred or ranked)[:maximum_genres]
        selected_genres = [entry[0] for entry in selected]
        readable_genres = _format_genres(selected_genres)
        template = PSEUDO_TEXT_TEMPLATES[
            _template_index(user_id, seed)
        ]
        rows.append(
            {
                "user_id": user_id,
                "user_index": user_index,
                "preferred_genres": "|".join(selected_genres),
                "used_fallback": used_fallback,
                "pseudo_text": template.format(
                    genres=readable_genres
                ),
            }
        )

    result = pd.DataFrame(rows).loc[
        :, list(USER_PSEUDO_TEXT_COLUMNS)
    ].astype(
        {
            "user_id": "int64",
            "user_index": "int64",
            "preferred_genres": "string",
            "used_fallback": "boolean",
            "pseudo_text": "string",
        }
    )
    if result["pseudo_text"].str.strip().eq("").any():
        raise TextFeatureError("generated user pseudo-text is empty")
    return result


def build_movie_texts(
    movies: pd.DataFrame,
    movie_mapping: IdMapping,
) -> pd.DataFrame:
    """Build one English title-and-genre document per catalog movie."""
    rows: list[dict[str, object]] = []
    for _, movie in movies.iterrows():
        movie_id = int(movie["movie_id"])
        selected_genres = [
            genre
            for genre in GENRE_COLUMNS
            if int(movie[genre]) == 1
        ]
        if not selected_genres:
            selected_genres = ["unknown"]
        readable_genres = _format_genres(selected_genres)
        rows.append(
            {
                "movie_id": movie_id,
                "movie_index": movie_mapping.external_to_index[movie_id],
                "genres": "|".join(selected_genres),
                "movie_text": (
                    f"{str(movie['title']).strip()}. "
                    f"Genres: {readable_genres}."
                ),
            }
        )

    result = pd.DataFrame(rows).loc[
        :, list(MOVIE_TEXT_COLUMNS)
    ].astype(
        {
            "movie_id": "int64",
            "movie_index": "int64",
            "genres": "string",
            "movie_text": "string",
        }
    ).sort_values("movie_index", ignore_index=True)
    if result["movie_text"].str.strip().eq("").any():
        raise TextFeatureError("generated movie text is empty")
    return result


def _encode_texts(
    texts: pd.Series,
    encoder: PreferenceTextEncoder,
) -> np.ndarray:
    """Encode text as a finite float32 matrix."""
    vectors = (
        encoder.encode_batch(texts.astype(str).tolist())
        .detach()
        .cpu()
        .numpy()
        .astype("float32", copy=False)
    )
    if vectors.ndim != 2 or vectors.shape[1] != TEXT_FEATURE_DIM:
        raise TextFeatureError(
            f"text vectors must have {TEXT_FEATURE_DIM} columns"
        )
    if not np.isfinite(vectors).all():
        raise TextFeatureError("text vectors contain non-finite values")
    return vectors


def build_text_feature_artifacts(
    train: pd.DataFrame,
    movies: pd.DataFrame,
    user_mapping: IdMapping,
    movie_mapping: IdMapping,
    *,
    data_version: str,
    feature_contract_version: str,
    positive_rating_threshold: float,
    maximum_genres: int,
    seed: int,
    minimum_genre_observations: int = 1,
    language: str = "en",
    generated_at: datetime | None = None,
) -> TextFeatureArtifacts:
    """Build source text, vectors and the versioned text contract."""
    if language != "en":
        raise TextFeatureError("generated project text must use English")
    _validate_sources(train, movies, user_mapping, movie_mapping)
    user_texts = build_user_pseudo_texts(
        train,
        movies,
        user_mapping,
        positive_rating_threshold=positive_rating_threshold,
        maximum_genres=maximum_genres,
        seed=seed,
        minimum_genre_observations=minimum_genre_observations,
    )
    movie_texts = build_movie_texts(movies, movie_mapping)
    encoder = PreferenceTextEncoder(
        TextEncoderConfig(dimension=TEXT_FEATURE_DIM)
    )
    user_vectors = _encode_texts(user_texts["pseudo_text"], encoder)
    movie_vectors = _encode_texts(movie_texts["movie_text"], encoder)

    timestamp = generated_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise TextFeatureError("generated_at must be timezone-aware")
    generated_at_utc = (
        timestamp.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    fallback_users = int(user_texts["used_fallback"].sum())
    preprocessor = {
        "schema_version": TEXT_FEATURE_SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc,
        "data_version": data_version,
        "feature_contract_version": feature_contract_version,
        "fit_partition": "train",
        "language": language,
        "seed": seed,
        "positive_rating_threshold": positive_rating_threshold,
        "maximum_genres": maximum_genres,
        "minimum_genre_observations": minimum_genre_observations,
        "selection_order": [
            "mean_rating_descending",
            "rating_count_descending",
            "fixed_genre_order",
        ],
        "excluded_user_preference_genres": ["unknown"],
        "fallback": "highest_mean_observed_genres",
        "templates": list(PSEUDO_TEXT_TEMPLATES),
        "movie_text_format": "{title}. Genres: {genres}.",
        "encoder": encoder.artifact_payload(),
        "text_feature_dimensions": TEXT_FEATURE_DIM,
        "text_fusion": TEXT_FUSION,
        "rows": {
            "users": len(user_texts),
            "movies": len(movie_texts),
            "fallback_users": fallback_users,
        },
        "limitations": [
            "Pseudo-text is generated from train ratings, not written by users.",
            "MovieLens 100K has no movie overview; movie text uses title and genres.",
            "Signed feature hashing is deterministic but not a semantic encoder.",
        ],
    }
    return TextFeatureArtifacts(
        user_texts=user_texts,
        movie_texts=movie_texts,
        user_vectors=user_vectors,
        movie_vectors=movie_vectors,
        preprocessor=preprocessor,
    )


def build_interaction_text_features(
    interactions: pd.DataFrame,
    artifacts: TextFeatureArtifacts,
) -> torch.Tensor:
    """Create one 128-column user-movie text interaction per row."""
    _require_exact_columns(
        interactions,
        MAPPED_RATING_COLUMNS,
        "interactions",
    )
    _validate_loaded_artifacts(artifacts)
    user_indices = interactions["user_index"].to_numpy(dtype="int64")
    movie_indices = interactions["movie_index"].to_numpy(dtype="int64")
    if user_indices.min() < 0 or user_indices.max() >= len(
        artifacts.user_vectors
    ):
        raise TextFeatureError("interaction contains invalid user_index")
    if movie_indices.min() < 0 or movie_indices.max() >= len(
        artifacts.movie_vectors
    ):
        raise TextFeatureError("interaction contains invalid movie_index")

    expected_user_ids = artifacts.user_texts.set_index("user_index").loc[
        user_indices, "user_id"
    ].to_numpy(dtype="int64")
    expected_movie_ids = artifacts.movie_texts.set_index("movie_index").loc[
        movie_indices, "movie_id"
    ].to_numpy(dtype="int64")
    if not np.array_equal(
        expected_user_ids,
        interactions["user_id"].to_numpy(dtype="int64"),
    ):
        raise TextFeatureError("interaction user ID/index mapping drifted")
    if not np.array_equal(
        expected_movie_ids,
        interactions["movie_id"].to_numpy(dtype="int64"),
    ):
        raise TextFeatureError("interaction movie ID/index mapping drifted")

    values = (
        artifacts.user_vectors[user_indices]
        * artifacts.movie_vectors[movie_indices]
    ).astype("float32", copy=False)
    if not np.isfinite(values).all():
        raise TextFeatureError(
            "interaction text features contain non-finite values"
        )
    return torch.from_numpy(values.copy())


def _save_vector_archive(
    path: Path,
    index_name: str,
    size: int,
    vectors: np.ndarray,
) -> None:
    """Write contiguous indices and vectors without pickle data."""
    np.savez_compressed(
        path,
        **{
            index_name: np.arange(size, dtype="int64"),
            "vectors": vectors.astype("float32", copy=False),
        },
    )


def save_text_feature_artifacts(
    artifacts: TextFeatureArtifacts,
    user_text_path: Path,
    movie_text_path: Path,
    user_vectors_path: Path,
    movie_vectors_path: Path,
    preprocessor_path: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    """Save text source tables, vector matrices and their contract."""
    _validate_loaded_artifacts(artifacts)
    paths = tuple(
        path.expanduser().resolve()
        for path in (
            user_text_path,
            movie_text_path,
            user_vectors_path,
            movie_vectors_path,
            preprocessor_path,
        )
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.user_texts.to_csv(paths[0], index=False)
    artifacts.movie_texts.to_csv(paths[1], index=False)
    _save_vector_archive(
        paths[2],
        "user_indices",
        len(artifacts.user_texts),
        artifacts.user_vectors,
    )
    _save_vector_archive(
        paths[3],
        "movie_indices",
        len(artifacts.movie_texts),
        artifacts.movie_vectors,
    )
    paths[4].write_text(
        json.dumps(artifacts.preprocessor, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return paths


def _load_vector_archive(
    path: Path,
    index_name: str,
) -> np.ndarray:
    """Load a safe NumPy archive and require contiguous row indices."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {index_name, "vectors"}:
                raise TextFeatureError(
                    f"{path.name} has unexpected array keys"
                )
            indices = archive[index_name]
            vectors = archive["vectors"]
    except (OSError, ValueError) as error:
        raise TextFeatureError(
            f"could not load text vector artifact: {path}"
        ) from error
    if not np.array_equal(indices, np.arange(len(indices))):
        raise TextFeatureError(f"{index_name} must be contiguous")
    if vectors.dtype != np.float32:
        raise TextFeatureError(f"{path.name} vectors must be float32")
    return vectors


def _validate_loaded_artifacts(
    artifacts: TextFeatureArtifacts,
) -> None:
    """Validate text tables, vector shapes and versioned metadata."""
    _require_exact_columns(
        artifacts.user_texts,
        USER_PSEUDO_TEXT_COLUMNS,
        "user pseudo-text",
    )
    _require_exact_columns(
        artifacts.movie_texts,
        MOVIE_TEXT_COLUMNS,
        "movie text",
    )
    expected_user_indices = np.arange(len(artifacts.user_texts))
    expected_movie_indices = np.arange(len(artifacts.movie_texts))
    if not np.array_equal(
        artifacts.user_texts["user_index"].to_numpy(),
        expected_user_indices,
    ):
        raise TextFeatureError("user text rows must follow user_index order")
    if not np.array_equal(
        artifacts.movie_texts["movie_index"].to_numpy(),
        expected_movie_indices,
    ):
        raise TextFeatureError(
            "movie text rows must follow movie_index order"
        )
    if artifacts.user_texts["user_id"].duplicated().any():
        raise TextFeatureError("user text contains duplicate user IDs")
    if artifacts.movie_texts["movie_id"].duplicated().any():
        raise TextFeatureError("movie text contains duplicate movie IDs")
    if artifacts.user_texts["pseudo_text"].isna().any() or (
        artifacts.user_texts["pseudo_text"].str.strip() == ""
    ).any():
        raise TextFeatureError("user pseudo-text must not be empty")
    if artifacts.movie_texts["movie_text"].isna().any() or (
        artifacts.movie_texts["movie_text"].str.strip() == ""
    ).any():
        raise TextFeatureError("movie text must not be empty")
    expected_shapes = (
        (len(artifacts.user_texts), TEXT_FEATURE_DIM),
        (len(artifacts.movie_texts), TEXT_FEATURE_DIM),
    )
    if artifacts.user_vectors.shape != expected_shapes[0]:
        raise TextFeatureError("user text vector shape is invalid")
    if artifacts.movie_vectors.shape != expected_shapes[1]:
        raise TextFeatureError("movie text vector shape is invalid")
    if artifacts.user_vectors.dtype != np.float32:
        raise TextFeatureError("user text vectors must be float32")
    if artifacts.movie_vectors.dtype != np.float32:
        raise TextFeatureError("movie text vectors must be float32")
    if not np.isfinite(artifacts.user_vectors).all() or not np.isfinite(
        artifacts.movie_vectors
    ).all():
        raise TextFeatureError("text vectors must contain finite values")
    if artifacts.preprocessor.get("schema_version") != (
        TEXT_FEATURE_SCHEMA_VERSION
    ):
        raise TextFeatureError("unsupported text feature schema version")
    if artifacts.preprocessor.get("fit_partition") != "train":
        raise TextFeatureError("text features must be fit from train")
    if artifacts.preprocessor.get("text_feature_dimensions") != (
        TEXT_FEATURE_DIM
    ):
        raise TextFeatureError("text feature dimension does not match code")
    if artifacts.preprocessor.get("text_fusion") != TEXT_FUSION:
        raise TextFeatureError("text fusion does not match code")
    encoder = artifacts.preprocessor.get("encoder")
    if not isinstance(encoder, dict) or encoder.get("dimension") != (
        TEXT_FEATURE_DIM
    ):
        raise TextFeatureError("text encoder dimension does not match code")
    rows = artifacts.preprocessor.get("rows")
    if not isinstance(rows, dict) or rows.get("users") != len(
        artifacts.user_texts
    ) or rows.get("movies") != len(artifacts.movie_texts):
        raise TextFeatureError("text artifact row counts do not match data")


def load_text_feature_artifacts(
    user_text_path: Path,
    movie_text_path: Path,
    user_vectors_path: Path,
    movie_vectors_path: Path,
    preprocessor_path: Path,
    *,
    expected_data_version: str | None = None,
    expected_feature_contract_version: str | None = None,
) -> TextFeatureArtifacts:
    """Load and validate all generated text-feature artifacts.

    When the expected versions are given, an artifact generated from a
    different data version or feature contract is rejected instead of
    being silently mixed into the run.
    """
    paths = tuple(
        path.expanduser().resolve()
        for path in (
            user_text_path,
            movie_text_path,
            user_vectors_path,
            movie_vectors_path,
            preprocessor_path,
        )
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"text feature artifact was not found: {path}"
            )
    user_texts = pd.read_csv(
        paths[0],
        dtype={
            "user_id": "int64",
            "user_index": "int64",
            "preferred_genres": "string",
            "used_fallback": "boolean",
            "pseudo_text": "string",
        },
    )
    movie_texts = pd.read_csv(
        paths[1],
        dtype={
            "movie_id": "int64",
            "movie_index": "int64",
            "genres": "string",
            "movie_text": "string",
        },
    )
    try:
        preprocessor = json.loads(paths[4].read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TextFeatureError(
            "invalid text feature preprocessor JSON"
        ) from error
    if not isinstance(preprocessor, dict):
        raise TextFeatureError(
            "text feature preprocessor must be a JSON object"
        )
    artifacts = TextFeatureArtifacts(
        user_texts=user_texts,
        movie_texts=movie_texts,
        user_vectors=_load_vector_archive(paths[2], "user_indices"),
        movie_vectors=_load_vector_archive(paths[3], "movie_indices"),
        preprocessor=preprocessor,
    )
    _validate_loaded_artifacts(artifacts)
    if (
        expected_data_version is not None
        and preprocessor.get("data_version") != expected_data_version
    ):
        raise TextFeatureError(
            "Text artifact data_version "
            f"'{preprocessor.get('data_version')}' does not match "
            f"expected '{expected_data_version}'"
        )
    if (
        expected_feature_contract_version is not None
        and preprocessor.get("feature_contract_version")
        != expected_feature_contract_version
    ):
        raise TextFeatureError(
            "Text artifact feature_contract_version "
            f"'{preprocessor.get('feature_contract_version')}' does not "
            f"match expected '{expected_feature_contract_version}'"
        )
    return artifacts
