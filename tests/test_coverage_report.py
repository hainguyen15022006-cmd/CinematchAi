"""Tests for the feature coverage and fallback report."""

from datetime import datetime, timezone
from pathlib import Path

import json
import pytest

from cinematch.features.coverage_report import (
    build_feature_coverage_report,
    save_feature_coverage_report,
)
from cinematch.features.numeric_features import (
    build_numeric_feature_artifacts,
)
from cinematch.features.pseudo_text import build_text_feature_artifacts
from tests.test_numeric_features import make_numeric_feature_fixture


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_fixture_report():
    train, movies, user_mapping, movie_mapping = (
        make_numeric_feature_fixture()
    )
    numeric = build_numeric_feature_artifacts(
        train,
        movies,
        user_mapping,
        movie_mapping,
        data_version="fixture-v1",
        feature_contract_version="hybrid-v1-167",
    )
    text = build_text_feature_artifacts(
        train,
        movies,
        user_mapping,
        movie_mapping,
        data_version="fixture-v1",
        feature_contract_version="hybrid-v1-167",
        positive_rating_threshold=4.0,
        maximum_genres=3,
        seed=42,
    )
    return build_feature_coverage_report(
        train,
        movies,
        numeric,
        text,
        data_version="fixture-v1",
        generated_at=datetime(2026, 9, 3, 8, 0, tzinfo=timezone.utc),
    ), train, movies, text


def test_coverage_report_counts_match_fixture_artifacts() -> None:
    report, train, movies, text = build_fixture_report()

    assert report["users"]["total"] == 3
    assert report["users"]["pseudo_text_fallback"] == int(
        text.user_texts["used_fallback"].sum()
    )
    assert report["movies"]["total"] == len(movies)
    assert report["movies"]["missing_release_year_imputed"] == 1
    assert report["movies"]["not_in_train"] == 1
    assert report["text_vectors"]["all_unit_norm"] is True
    assert (
        report["normalized_release_year"]["policy"]
        == "report_only_no_clipping"
    )


def test_coverage_report_round_trip(tmp_path) -> None:
    report, _, _, _ = build_fixture_report()
    path = save_feature_coverage_report(
        report,
        tmp_path / "feature_coverage_report.json",
    )
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded == report
    assert loaded["schema_version"] == "1.0"
    assert loaded["generated_at_utc"].endswith("Z")


def test_real_coverage_report_matches_artifacts_if_available() -> None:
    report_path = (
        PROJECT_ROOT
        / "outputs"
        / "features"
        / "feature_coverage_report.json"
    )
    pseudo_path = (
        PROJECT_ROOT / "outputs" / "features" / "user_pseudo_text.csv"
    )
    if not (report_path.exists() and pseudo_path.exists()):
        pytest.skip("Real coverage report is not available")

    import pandas as pd

    report = json.loads(report_path.read_text(encoding="utf-8"))
    pseudo = pd.read_csv(pseudo_path)

    assert report["users"]["total"] == len(pseudo)
    assert report["users"]["pseudo_text_fallback"] == int(
        pseudo["used_fallback"].sum()
    )
