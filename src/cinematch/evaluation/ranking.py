"""Ranking metrics for evaluating movie recommendations."""
import math
from collections.abc import Collection, Sequence


def _validate_k(k: int) -> None:
    """Validate the recommendation cutoff."""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")


def _validate_recommended_items(
    recommended_items: Sequence[int],
) -> tuple[int, ...]:
    """Validate and normalize a ranked recommendation list."""

    ranking = tuple(recommended_items)

    if len(ranking) != len(set(ranking)):
        raise ValueError("Recommended items must not contain duplicates")

    return ranking


def recall_at_k(
    recommended_items: Sequence[int],
    relevant_items: Collection[int],
    k: int = 10,
) -> float:
    """Calculate the fraction of relevant items retrieved in the top K.

    Args:
        recommended_items:
            Movie identifiers ordered from most to least recommended.
        relevant_items:
            Movie identifiers considered relevant in the test data.
        k:
            Number of top-ranked recommendations to evaluate.

    Returns:
        Recall as a value from 0.0 to 1.0. If there are no relevant
        items, the function returns 0.0.

    Raises:
        ValueError:
            If k is invalid or the recommendation list contains
            duplicate movie identifiers.
    """

    _validate_k(k)
    ranking = _validate_recommended_items(recommended_items)
    relevant = set(relevant_items)

    if not relevant:
        return 0.0

    top_k_items = set(ranking[:k])
    number_of_hits = len(top_k_items.intersection(relevant))

    return number_of_hits / len(relevant)

def hit_rate_at_k(
    recommended_items: Sequence[int],
    relevant_items: Collection[int],
    k: int = 10,
) -> float:
    """Check whether the top K contains at least one relevant item.

    Args:
        recommended_items:
            Movie identifiers ordered from most to least recommended.
        relevant_items:
            Movie identifiers considered relevant in the test data.
        k:
            Number of top-ranked recommendations to evaluate.

    Returns:
        1.0 if at least one relevant movie appears in the top K;
        otherwise, 0.0.

    Raises:
        ValueError:
            If k is invalid or the recommendation list contains
            duplicate movie identifiers.
    """

    _validate_k(k)
    ranking = _validate_recommended_items(recommended_items)
    relevant = set(relevant_items)

    top_k_items = set(ranking[:k])
    has_relevant_item = bool(top_k_items.intersection(relevant))

    return float(has_relevant_item)



def ndcg_at_k(
    recommended_items: Sequence[int],
    relevant_items: Collection[int],
    k: int = 10,
) -> float:
    """Calculate binary Normalized Discounted Cumulative Gain at K.

    Relevant movies receive a relevance value of 1. Non-relevant
    movies receive a relevance value of 0. Relevant movies appearing
    near the beginning of the ranking contribute more to the score.

    Args:
        recommended_items:
            Movie identifiers ordered from most to least recommended.
        relevant_items:
            Movie identifiers considered relevant in the test data.
        k:
            Number of top-ranked recommendations to evaluate.

    Returns:
        NDCG as a value from 0.0 to 1.0. If there are no relevant
        items, the function returns 0.0.

    Raises:
        ValueError:
            If k is invalid or the recommendation list contains
            duplicate movie identifiers.
    """

    _validate_k(k)
    ranking = _validate_recommended_items(recommended_items)
    relevant = set(relevant_items)

    if not relevant:
        return 0.0

    dcg = 0.0

    for rank, movie_id in enumerate(ranking[:k], start=1):
        if movie_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_number_of_hits = min(len(relevant), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_number_of_hits + 1)
    )

    return dcg / idcg



def coverage_at_k(
    recommendation_lists: Sequence[Sequence[int]],
    catalog_items: Collection[int],
    k: int = 10,
) -> float:
    """Calculate catalog coverage across recommendation lists.

    Coverage is the fraction of eligible catalog movies appearing
    in at least one top-K recommendation list.

    Args:
        recommendation_lists:
            Ranked movie identifiers for multiple users or groups.
        catalog_items:
            All movie identifiers eligible for recommendation.
        k:
            Number of top-ranked items used from each list.

    Returns:
        Coverage as a value from 0.0 to 1.0.

    Raises:
        ValueError:
            If k is invalid, the catalog is empty, a recommendation
            list contains duplicates, or a recommended movie does
            not belong to the catalog.
    """

    _validate_k(k)
    catalog = set(catalog_items)

    if not catalog:
        raise ValueError("Catalog items must not be empty")

    covered_items: set[int] = set()

    for recommended_items in recommendation_lists:
        ranking = _validate_recommended_items(recommended_items)
        unknown_items = set(ranking).difference(catalog)

        if unknown_items:
            raise ValueError(
                "Recommended items must belong to the catalog"
            )

        covered_items.update(ranking[:k])

    return len(covered_items) / len(catalog)