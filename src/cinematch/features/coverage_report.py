"""Coverage and fallback report for the Hybrid feature artifacts."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cinematch.data.schema import GENRE_COLUMNS
from cinematch.features.numeric_features import NumericFeatureArtifacts
from cinematch.features.pseudo_text import TextFeatureArtifacts

COVERAGE_REPORT_SCHEMA_VERSION = "1.0"
NORMALIZED_YEAR_WARNING_LIMIT = 6.0


def build_feature_coverage_report(
    train: pd.DataFrame,
    movies: pd.DataFrame,
    numeric_artifacts: NumericFeatureArtifacts,
    text_artifacts: TextFeatureArtifacts,
    *,
    data_version: str,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Summarize how well the feature artifacts cover users and movies.

    The report only reads existing artifacts; it never recomputes or
    changes a feature. Its purpose is to make fallback and imputation
    behaviour visible in the data report instead of hidden in code.
    """
    timestamp = generated_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("generated_at must be timezone-aware")

    user_texts = text_artifacts.user_texts
    fallback_count = int(user_texts["used_fallback"].sum())
    genre_counts = Counter(
        genre
        for joined in user_texts["preferred_genres"]
        for genre in str(joined).split("|")
    )
    preference_genres = [g for g in GENRE_COLUMNS if g != "unknown"]
    never_selected = [
        genre for genre in preference_genres if genre not in genre_counts
    ]

    profiles = numeric_artifacts.user_profiles
    history_columns = [
        column
        for column in profiles.columns
        if column not in ("user_id", "user_index")
    ]
    empty_history_users = int(
        (profiles[history_columns].abs().sum(axis=1) == 0.0).sum()
    )

    train_movie_ids = set(train["movie_id"])
    movies_not_in_train = int(
        (~movies["movie_id"].isin(train_movie_ids)).sum()
    )
    movies_missing_year = int(movies["release_year"].isna().sum())

    movie_features = numeric_artifacts.movie_features
    year_column = "normalized_release_year"
    year_values = movie_features[year_column].astype(float)
    year_minimum = float(year_values.min())
    year_maximum = float(year_values.max())
    year_outliers = int(
        (year_values.abs() > NORMALIZED_YEAR_WARNING_LIMIT).sum()
    )

    user_norms = np.linalg.norm(text_artifacts.user_vectors, axis=1)
    movie_norms = np.linalg.norm(text_artifacts.movie_vectors, axis=1)

    return {
        "schema_version": COVERAGE_REPORT_SCHEMA_VERSION,
        "generated_at_utc": (
            timestamp.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        ),
        "data_version": data_version,
        "artifact_type": "feature_coverage_report",
        "users": {
            "total": len(user_texts),
            "pseudo_text_fallback": fallback_count,
            "pseudo_text_fallback_share": round(
                fallback_count / len(user_texts), 4
            ),
            "empty_history_profiles": empty_history_users,
        },
        "pseudo_text_genres": {
            "selection_counts": dict(
                sorted(
                    genre_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "never_selected": never_selected,
        },
        "movies": {
            "total": len(movies),
            "missing_release_year_imputed": movies_missing_year,
            "not_in_train": movies_not_in_train,
        },
        "normalized_release_year": {
            "minimum": round(year_minimum, 4),
            "maximum": round(year_maximum, 4),
            "warning_limit": NORMALIZED_YEAR_WARNING_LIMIT,
            "outliers_beyond_limit": year_outliers,
            "policy": "report_only_no_clipping",
        },
        "text_vectors": {
            "user_rows": int(text_artifacts.user_vectors.shape[0]),
            "movie_rows": int(text_artifacts.movie_vectors.shape[0]),
            "all_unit_norm": bool(
                np.allclose(user_norms, 1.0, atol=1e-5)
                and np.allclose(movie_norms, 1.0, atol=1e-5)
            ),
        },
    }


def save_feature_coverage_report(
    report: dict[str, Any],
    path: Path,
) -> Path:
    """Write the coverage report as pretty-printed JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination
