from pathlib import Path

import pandas as pd
import pytest

from cinematch.data.io import load_ml100k_ratings
from cinematch.data.profiling import (
    build_ratings_profile,
)
from cinematch.data.schema import RATING_COLUMNS
from cinematch.data.validation import (
    DataValidationError,
    validate_ratings,
)


def make_valid_ratings() -> pd.DataFrame:
    """Create a small deterministic rating fixture."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "movie_id": [10, 20, 10, 30],
            "rating": [5.0, 4.0, 3.0, 2.0],
            "timestamp": [100, 200, 150, 250],
        }
    ).astype(
        {
            "user_id": "int64",
            "movie_id": "int64",
            "rating": "float32",
            "timestamp": "int64",
        }
    )


def test_load_ml100k_ratings(
    tmp_path: Path,
) -> None:
    ratings_path = tmp_path / "u.data"

    ratings_path.write_text(
        "1\t10\t5\t100\n"
        "2\t20\t4\t200\n",
        encoding="utf-8",
    )

    ratings = load_ml100k_ratings(
        ratings_path
    )

    assert tuple(ratings.columns) == RATING_COLUMNS
    assert len(ratings) == 2
    assert ratings["rating"].tolist() == [5.0, 4.0]


def test_missing_file_raises_clear_error(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.data"

    with pytest.raises(
        FileNotFoundError,
        match="ratings file was not found",
    ):
        load_ml100k_ratings(missing_path)


def test_valid_ratings_pass_validation() -> None:
    ratings = make_valid_ratings()

    validate_ratings(ratings)


def test_rating_outside_range_is_rejected() -> None:
    ratings = make_valid_ratings()
    ratings.loc[0, "rating"] = 7.0

    with pytest.raises(
        DataValidationError,
        match="outside",
    ):
        validate_ratings(ratings)


def test_missing_value_is_rejected() -> None:
    ratings = make_valid_ratings()
    ratings.loc[0, "rating"] = None

    with pytest.raises(
        DataValidationError,
        match="missing",
    ):
        validate_ratings(ratings)


def test_profile_has_expected_values() -> None:
    ratings = make_valid_ratings()

    validate_ratings(ratings)
    profile = build_ratings_profile(ratings)

    assert profile.number_of_ratings == 4
    assert profile.number_of_users == 2
    assert profile.number_of_movies == 3

    assert profile.minimum_rating == 2.0
    assert profile.maximum_rating == 5.0
    assert profile.mean_rating == pytest.approx(3.5)

    assert profile.rating_distribution == {
        2: 1,
        3: 1,
        4: 1,
        5: 1,
    }

    assert profile.possible_interactions == 6
    assert profile.density == pytest.approx(4 / 6)
    assert profile.sparsity == pytest.approx(2 / 6)

def test_real_ml100k_profile_if_available() -> None:
    ratings_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "raw"
        / "ml-100k"
        / "u.data"
    )

    if not ratings_path.exists():
        pytest.skip("MovieLens 100K is not downloaded")

    ratings = load_ml100k_ratings(ratings_path)

    validate_ratings(ratings)
    profile = build_ratings_profile(ratings)

    assert profile.number_of_ratings == 100_000
    assert profile.number_of_users == 943
    assert profile.number_of_movies == 1_682
    assert profile.minimum_rating == 1.0
    assert profile.maximum_rating == 5.0
    assert profile.exact_duplicate_rows == 0
    assert profile.duplicate_user_movie_rows == 0