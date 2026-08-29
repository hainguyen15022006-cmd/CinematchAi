

"""Tests for offline evaluation candidate construction."""

import json
from pathlib import Path

import pandas as pd
import pytest

from cinematch.evaluation import (
    build_candidate_set,
    build_positive_items,
    build_seen_items,
    sample_negative_items,
)

def test_build_seen_items_combines_train_and_validation() -> None:
    """Seen items must include both train and validation movies."""

    result = build_seen_items(
        train_movie_indices=[10, 20, 30],
        validation_movie_indices=[30, 40],
    )

    assert result == frozenset({10, 20, 30, 40})


def test_build_seen_items_accepts_empty_validation() -> None:
    """A user may have no validation movies in a small test example."""

    result = build_seen_items(
        train_movie_indices=[10, 20],
        validation_movie_indices=[],
    )

    assert result == frozenset({10, 20})


def test_build_seen_items_removes_duplicate_indices() -> None:
    """Repeated input indices must appear only once."""

    result = build_seen_items(
        train_movie_indices=[10, 10, 20],
        validation_movie_indices=[20, 20, 30],
    )

    assert result == frozenset({10, 20, 30})


@pytest.mark.parametrize(
    ("train_items", "validation_items"),
    [
        ([-1, 10], [20]),
        ([10], [-1, 20]),
    ],
)
def test_build_seen_items_rejects_negative_indices(
    train_items: list[int],
    validation_items: list[int],
) -> None:
    """Movie indices must use the non-negative mapping range."""

    with pytest.raises(
        ValueError,
        match="non-negative movie indices",
    ):
        build_seen_items(
            train_movie_indices=train_items,
            validation_movie_indices=validation_items,
        )



def test_build_positive_items_uses_shared_threshold() -> None:
    """Ratings equal to or above 4.0 must be positive."""

    result = build_positive_items(
        test_movie_indices=[10, 20, 30],
        test_ratings=[5.0, 3.0, 4.0],
    )

    assert result == frozenset({10, 30})


def test_build_positive_items_returns_empty_without_positive_ratings() -> None:
    """A user may have no relevant movies in the test partition."""

    result = build_positive_items(
        test_movie_indices=[10, 20, 30],
        test_ratings=[1.0, 2.0, 3.0],
    )

    assert result == frozenset()


def test_build_positive_items_accepts_a_custom_threshold() -> None:
    """The threshold can be changed explicitly for an experiment."""

    result = build_positive_items(
        test_movie_indices=[10, 20, 30],
        test_ratings=[3.0, 3.5, 4.0],
        positive_threshold=3.5,
    )

    assert result == frozenset({20, 30})


def test_build_positive_items_rejects_different_input_lengths() -> None:
    """Every test movie must have exactly one corresponding rating."""

    with pytest.raises(
        ValueError,
        match="same number of values",
    ):
        build_positive_items(
            test_movie_indices=[10, 20],
            test_ratings=[5.0],
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        float("nan"),
        0.5,
        5.5,
    ],
)
def test_build_positive_items_rejects_invalid_threshold(
    invalid_threshold: float,
) -> None:
    """Positive threshold must remain on the MovieLens rating scale."""

    with pytest.raises(
        ValueError,
        match="positive_threshold",
    ):
        build_positive_items(
            test_movie_indices=[10],
            test_ratings=[5.0],
            positive_threshold=invalid_threshold,
        )


@pytest.mark.parametrize(
    "invalid_rating",
    [
        float("nan"),
        0.0,
        6.0,
    ],
)
def test_build_positive_items_rejects_invalid_ratings(
    invalid_rating: float,
) -> None:
    """Test ratings must be finite and between one and five."""

    with pytest.raises(
        ValueError,
        match="test_ratings",
    ):
        build_positive_items(
            test_movie_indices=[10],
            test_ratings=[invalid_rating],
        )


def test_build_positive_items_rejects_duplicate_movie_indices() -> None:
    """One user must not have duplicate movies in the test input."""

    with pytest.raises(
        ValueError,
        match="must not contain duplicates",
    ):
        build_positive_items(
            test_movie_indices=[10, 10],
            test_ratings=[4.0, 5.0],
        )



def test_sample_negative_items_only_uses_eligible_movies() -> None:
    """Sampled negatives must exclude seen and positive movies."""

    result = sample_negative_items(
        catalog_movie_indices=range(10),
        seen_items={0, 1, 2},
        positive_items={8},
        number_of_negatives=4,
        seed=42,
    )

    assert len(result) == 4
    assert len(set(result)) == 4
    assert set(result).issubset({3, 4, 5, 6, 7, 9})


def test_sample_negative_items_is_reproducible() -> None:
    """The same seed and inputs must produce the same sample."""

    first_result = sample_negative_items(
        catalog_movie_indices=range(100),
        seen_items={0, 1},
        positive_items={98, 99},
        number_of_negatives=10,
        seed=42,
    )
    second_result = sample_negative_items(
        catalog_movie_indices=range(100),
        seen_items={0, 1},
        positive_items={98, 99},
        number_of_negatives=10,
        seed=42,
    )

    assert first_result == second_result


def test_sample_negative_items_changes_with_seed() -> None:
    """Different seeds should produce different negative samples."""

    first_result = sample_negative_items(
        catalog_movie_indices=range(100),
        seen_items={0, 1},
        positive_items={98, 99},
        number_of_negatives=10,
        seed=42,
    )
    second_result = sample_negative_items(
        catalog_movie_indices=range(100),
        seen_items={0, 1},
        positive_items={98, 99},
        number_of_negatives=10,
        seed=43,
    )

    assert first_result != second_result


def test_sample_negative_items_rejects_insufficient_candidates() -> None:
    """The function must not silently return fewer negatives."""

    with pytest.raises(
        ValueError,
        match="Not enough eligible",
    ):
        sample_negative_items(
            catalog_movie_indices={0, 1, 2},
            seen_items={0, 1},
            positive_items={2},
            number_of_negatives=1,
        )


def test_sample_negative_items_rejects_seen_positive_overlap() -> None:
    """A positive test item must not already be marked as seen."""

    with pytest.raises(
        ValueError,
        match="must not overlap",
    ):
        sample_negative_items(
            catalog_movie_indices=range(10),
            seen_items={0, 1, 2},
            positive_items={2, 8},
            number_of_negatives=2,
        )


def test_sample_negative_items_rejects_items_outside_catalog() -> None:
    """Seen and positive items must belong to the catalog."""

    with pytest.raises(
        ValueError,
        match="must belong to the catalog",
    ):
        sample_negative_items(
            catalog_movie_indices=range(10),
            seen_items={0, 99},
            positive_items={8},
            number_of_negatives=2,
        )


@pytest.mark.parametrize("invalid_count", [-1, True])
def test_sample_negative_items_rejects_invalid_count(
    invalid_count: object,
) -> None:
    """The requested negative count must be a non-negative integer."""

    with pytest.raises(
        ValueError,
        match="non-negative integer",
    ):
        sample_negative_items(
            catalog_movie_indices=range(10),
            seen_items={0},
            positive_items={9},
            number_of_negatives=invalid_count,  # type: ignore[arg-type]
        )


def test_sample_negative_items_rejects_invalid_seed() -> None:
    """The reproducibility seed must be an integer."""

    with pytest.raises(
        ValueError,
        match="seed must be an integer",
    ):
        sample_negative_items(
            catalog_movie_indices=range(10),
            seen_items={0},
            positive_items={9},
            number_of_negatives=2,
            seed=True,
        )



def test_build_candidate_set_keeps_every_positive_item() -> None:
    """All relevant test movies must remain in the candidates."""

    positives = {18, 19}

    result = build_candidate_set(
        catalog_movie_indices=range(20),
        seen_items={0, 1, 2},
        positive_items=positives,
        number_of_negatives=5,
        seed=42,
    )

    assert positives.issubset(result)
    assert len(result) == 7
    assert len(set(result)) == 7


def test_build_candidate_set_excludes_seen_items() -> None:
    """Train and validation movies must never become candidates."""

    seen = {0, 1, 2, 3}

    result = build_candidate_set(
        catalog_movie_indices=range(20),
        seen_items=seen,
        positive_items={19},
        number_of_negatives=5,
        seed=42,
    )

    assert set(result).isdisjoint(seen)


def test_build_candidate_set_is_reproducible() -> None:
    """The same inputs and seed must produce identical candidates."""

    first_result = build_candidate_set(
        catalog_movie_indices=range(100),
        seen_items={0, 1},
        positive_items={98, 99},
        number_of_negatives=10,
        seed=42,
    )
    second_result = build_candidate_set(
        catalog_movie_indices=range(100),
        seen_items={0, 1},
        positive_items={98, 99},
        number_of_negatives=10,
        seed=42,
    )

    assert first_result == second_result


def test_build_candidate_set_uses_default_negative_count() -> None:
    """The shared protocol must sample 100 negatives by default."""

    result = build_candidate_set(
        catalog_movie_indices=range(200),
        seen_items={0},
        positive_items={199},
        seed=42,
    )

    assert len(result) == 101


def test_build_candidate_set_rejects_user_without_positives() -> None:
    """Users without positive test items are not metric-eligible."""

    with pytest.raises(
        ValueError,
        match="at least one positive",
    ):
        build_candidate_set(
            catalog_movie_indices=range(20),
            seen_items={0, 1},
            positive_items=set(),
            number_of_negatives=5,
            seed=42,
        )




def test_real_movielens_candidate_protocol_if_available() -> None:
    """Candidate construction must satisfy the real data contract."""

    project_root = Path(__file__).resolve().parents[1]
    processed_directory = project_root / "data" / "processed"

    train_path = processed_directory / "train.csv"
    validation_path = processed_directory / "validation.csv"
    test_path = processed_directory / "test.csv"
    mappings_path = processed_directory / "id_mappings.json"

    required_paths = [
        train_path,
        validation_path,
        test_path,
        mappings_path,
    ]

    if not all(path.exists() for path in required_paths):
        pytest.skip("Processed MovieLens data is not available")

    train = pd.read_csv(
        train_path,
        usecols=["user_index", "movie_index"],
    )
    validation = pd.read_csv(
        validation_path,
        usecols=["user_index", "movie_index"],
    )
    test = pd.read_csv(
        test_path,
        usecols=["user_index", "movie_index", "rating"],
    )

    with mappings_path.open(
        "r",
        encoding="utf-8",
    ) as mapping_file:
        mappings = json.load(mapping_file)

    number_of_movies = len(
        mappings["movies"]["external_ids"]
    )
    catalog_movie_indices = range(number_of_movies)

    train_by_user = (
        train.groupby("user_index")["movie_index"]
        .apply(list)
        .to_dict()
    )
    validation_by_user = (
        validation.groupby("user_index")["movie_index"]
        .apply(list)
        .to_dict()
    )

    eligible_user_count = 0
    skipped_user_count = 0
    first_candidate_set: tuple[int, ...] | None = None
    first_candidate_inputs: tuple[
        frozenset[int],
        frozenset[int],
    ] | None = None

    for user_index, user_test in test.groupby(
        "user_index",
        sort=True,
    ):
        seen_items = build_seen_items(
            train_movie_indices=train_by_user.get(
                user_index,
                [],
            ),
            validation_movie_indices=validation_by_user.get(
                user_index,
                [],
            ),
        )
        positive_items = build_positive_items(
            test_movie_indices=user_test["movie_index"].tolist(),
            test_ratings=user_test["rating"].tolist(),
            positive_threshold=4.0,
        )

        if not positive_items:
            skipped_user_count += 1
            continue

        eligible_user_count += 1

        candidate_set = build_candidate_set(
            catalog_movie_indices=catalog_movie_indices,
            seen_items=seen_items,
            positive_items=positive_items,
            number_of_negatives=100,
            seed=42,
        )

        assert positive_items.issubset(candidate_set)
        assert set(candidate_set).isdisjoint(seen_items)
        assert len(candidate_set) == len(positive_items) + 100
        assert len(candidate_set) == len(set(candidate_set))

        if first_candidate_set is None:
            first_candidate_set = candidate_set
            first_candidate_inputs = (
                seen_items,
                positive_items,
            )

    assert eligible_user_count == 836
    assert skipped_user_count == 107
    assert eligible_user_count + skipped_user_count == 943

    assert first_candidate_set is not None
    assert first_candidate_inputs is not None

    repeated_candidate_set = build_candidate_set(
        catalog_movie_indices=catalog_movie_indices,
        seen_items=first_candidate_inputs[0],
        positive_items=first_candidate_inputs[1],
        number_of_negatives=100,
        seed=42,
    )

    assert repeated_candidate_set == first_candidate_set