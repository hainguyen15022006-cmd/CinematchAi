"""Candidate construction for offline recommendation evaluation."""


import math
import random
from collections.abc import Iterable
from numbers import Integral, Real
DEFAULT_RANDOM_SEED = 42
DEFAULT_NEGATIVE_SAMPLE_SIZE = 100

def _normalize_movie_indices(
    movie_indices: Iterable[int],
    source_name: str,
) -> frozenset[int]:
    """Validate movie indices and return their unique values."""

    normalized_indices: set[int] = set()

    for movie_index in movie_indices:
        if isinstance(movie_index, bool) or not isinstance(
            movie_index,
            Integral,
        ):
            raise ValueError(
                f"{source_name} must contain integer movie indices"
            )

        normalized_index = int(movie_index)

        if normalized_index < 0:
            raise ValueError(
                f"{source_name} must contain non-negative movie indices"
            )

        normalized_indices.add(normalized_index)

    return frozenset(normalized_indices)


def build_seen_items(
    train_movie_indices: Iterable[int],
    validation_movie_indices: Iterable[int],
) -> frozenset[int]:
    """Build the set of movies observed before test evaluation.

    Train and validation interactions are considered seen. Test
    interactions must not be passed to this function because they
    are held out for final evaluation.

    Args:
        train_movie_indices:
            Movie indices from one user's training interactions.
        validation_movie_indices:
            Movie indices from the same user's validation interactions.

    Returns:
        Unique movie indices observed in train or validation.

    Raises:
        ValueError:
            If an input contains a non-integer or negative movie index.
    """

    train_items = _normalize_movie_indices(
        train_movie_indices,
        "train_movie_indices",
    )
    validation_items = _normalize_movie_indices(
        validation_movie_indices,
        "validation_movie_indices",
    )

    return train_items.union(validation_items)



def build_positive_items(
    test_movie_indices: Iterable[int],
    test_ratings: Iterable[float],
    *,
    positive_threshold: float = 4.0,
) -> frozenset[int]:
    """Build relevant test items using the positive rating threshold.

    Args:
        test_movie_indices:
            Movie indices from one user's held-out test interactions.
        test_ratings:
            Ratings corresponding to the test movie indices.
        positive_threshold:
            Minimum rating considered relevant. The shared CineMatch
            data contract uses 4.0.

    Returns:
        Unique movie indices whose test ratings meet or exceed the
        positive threshold.

    Raises:
        ValueError:
            If indices and ratings have different lengths, movie
            indices are invalid or duplicated, ratings are invalid,
            or the threshold is outside the MovieLens rating scale.
    """

    raw_movie_indices = tuple(test_movie_indices)
    unique_movie_indices = _normalize_movie_indices(
        raw_movie_indices,
        "test_movie_indices",
    )
    raw_ratings = tuple(test_ratings)

    if len(raw_movie_indices) != len(raw_ratings):
        raise ValueError(
            "test_movie_indices and test_ratings must contain "
            "the same number of values"
        )

    if len(raw_movie_indices) != len(unique_movie_indices):
        raise ValueError(
            "test_movie_indices must not contain duplicates"
        )

    if (
        isinstance(positive_threshold, bool)
        or not isinstance(positive_threshold, Real)
        or not math.isfinite(float(positive_threshold))
        or not 1.0 <= float(positive_threshold) <= 5.0
    ):
        raise ValueError(
            "positive_threshold must be a finite number "
            "between 1.0 and 5.0"
        )

    normalized_ratings: list[float] = []

    for rating in raw_ratings:
        if (
            isinstance(rating, bool)
            or not isinstance(rating, Real)
            or not math.isfinite(float(rating))
            or not 1.0 <= float(rating) <= 5.0
        ):
            raise ValueError(
                "test_ratings must contain finite values "
                "between 1.0 and 5.0"
            )

        normalized_ratings.append(float(rating))

    threshold = float(positive_threshold)

    return frozenset(
        int(movie_index)
        for movie_index, rating in zip(
            raw_movie_indices,
            normalized_ratings,
            strict=True,
        )
        if rating >= threshold
    )



def sample_negative_items(
    catalog_movie_indices: Iterable[int],
    seen_items: Iterable[int],
    positive_items: Iterable[int],
    *,
    number_of_negatives: int,
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[int, ...]:
    """Sample reproducible negative items from the eligible catalog.

    Eligible negatives belong to the catalog but are neither seen
    before evaluation nor relevant positive test items.

    Args:
        catalog_movie_indices:
            All movie indices eligible for recommendation.
        seen_items:
            Movies observed in train or validation.
        positive_items:
            Relevant movies from the held-out test partition.
        number_of_negatives:
            Exact number of negative items to sample.
        seed:
            Random seed used to reproduce the same sample.

    Returns:
        Sampled negative movie indices in deterministic random order.

    Raises:
        ValueError:
            If count or seed is invalid, positive and seen items
            overlap, an item is outside the catalog, or there are
            not enough eligible negatives.
    """

    if (
        isinstance(number_of_negatives, bool)
        or not isinstance(number_of_negatives, Integral)
        or int(number_of_negatives) < 0
    ):
        raise ValueError(
            "number_of_negatives must be a non-negative integer"
        )

    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")

    catalog = _normalize_movie_indices(
        catalog_movie_indices,
        "catalog_movie_indices",
    )
    seen = _normalize_movie_indices(
        seen_items,
        "seen_items",
    )
    positives = _normalize_movie_indices(
        positive_items,
        "positive_items",
    )

    unknown_items = seen.union(positives).difference(catalog)

    if unknown_items:
        raise ValueError(
            "Seen and positive items must belong to the catalog"
        )

    if seen.intersection(positives):
        raise ValueError(
            "Seen items and positive items must not overlap"
        )

    eligible_negatives = sorted(
        catalog.difference(seen).difference(positives)
    )
    requested_count = int(number_of_negatives)

    if requested_count > len(eligible_negatives):
        raise ValueError(
            "Not enough eligible negative items to sample"
        )

    random_generator = random.Random(int(seed))

    return tuple(
        random_generator.sample(
            eligible_negatives,
            requested_count,
        )
    )



def build_candidate_set(
    catalog_movie_indices: Iterable[int],
    seen_items: Iterable[int],
    positive_items: Iterable[int],
    *,
    number_of_negatives: int = DEFAULT_NEGATIVE_SAMPLE_SIZE,
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[int, ...]:
    """Build one reproducible evaluation candidate set.

    Every positive test item is retained. Negative items are sampled
    from catalog movies that are neither seen nor positive.

    Args:
        catalog_movie_indices:
            All movie indices eligible for evaluation.
        seen_items:
            Movies observed in train or validation.
        positive_items:
            Relevant movies from the held-out test partition.
        number_of_negatives:
            Number of non-relevant candidate movies to sample.
        seed:
            Random seed used for reproducible negative sampling.

    Returns:
        A sorted tuple containing all positives and sampled negatives.

    Raises:
        ValueError:
            If there is no positive item or candidate construction
            inputs violate the shared evaluation protocol.
    """

    positives = _normalize_movie_indices(
        positive_items,
        "positive_items",
    )

    if not positives:
        raise ValueError(
            "Candidate evaluation requires at least one positive item"
        )

    negative_items = sample_negative_items(
        catalog_movie_indices=catalog_movie_indices,
        seen_items=seen_items,
        positive_items=positives,
        number_of_negatives=number_of_negatives,
        seed=seed,
    )

    candidates = positives.union(negative_items)

    return tuple(sorted(candidates))