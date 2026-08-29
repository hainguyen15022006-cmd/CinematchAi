"""Tests for recommendation ranking metrics."""

import pytest

from cinematch.evaluation import (
    coverage_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
)

def test_recall_at_k_matches_manual_calculation() -> None:
    """Recall must match an example calculated by hand."""

    recommended_items = [10, 20, 30, 40, 50]
    relevant_items = {20, 40, 60}

    result = recall_at_k(
        recommended_items,
        relevant_items,
        k=3,
    )

    assert result == pytest.approx(1 / 3)


def test_recall_at_k_only_uses_top_k_items() -> None:
    """A relevant movie below position K must not count as a hit."""

    recommended_items = [10, 20, 30, 40]
    relevant_items = {40}

    result = recall_at_k(
        recommended_items,
        relevant_items,
        k=3,
    )

    assert result == 0.0


def test_recall_at_k_returns_one_when_all_relevant_items_are_found() -> None:
    """Recall must be one when every relevant movie is retrieved."""

    recommended_items = [10, 20, 30]
    relevant_items = {10, 30}

    result = recall_at_k(
        recommended_items,
        relevant_items,
        k=3,
    )

    assert result == 1.0


def test_recall_at_k_accepts_a_ranking_shorter_than_k() -> None:
    """Evaluation must work when fewer than K movies are available."""

    recommended_items = [10, 20]
    relevant_items = {10, 20, 30}

    result = recall_at_k(
        recommended_items,
        relevant_items,
        k=10,
    )

    assert result == pytest.approx(2 / 3)


def test_recall_at_k_returns_zero_without_relevant_items() -> None:
    """A case without relevant movies has zero recall."""

    result = recall_at_k(
        recommended_items=[10, 20, 30],
        relevant_items=set(),
        k=3,
    )

    assert result == 0.0


@pytest.mark.parametrize("invalid_k", [0, -1])
def test_recall_at_k_rejects_invalid_k(invalid_k: int) -> None:
    """K must be a positive integer."""

    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        recall_at_k(
            recommended_items=[10, 20],
            relevant_items={10},
            k=invalid_k,
        )


def test_recall_at_k_rejects_duplicate_recommendations() -> None:
    """A ranked recommendation list must not repeat a movie."""

    with pytest.raises(
        ValueError,
        match="must not contain duplicates",
    ):
        recall_at_k(
            recommended_items=[10, 20, 10],
            relevant_items={10},
            k=3,
        )



def test_hit_rate_at_k_returns_one_when_a_relevant_item_is_found() -> None:
    """Hit Rate must be one when the top K contains a relevant movie."""

    recommended_items = [10, 20, 30, 40]
    relevant_items = {20, 50}

    result = hit_rate_at_k(
        recommended_items,
        relevant_items,
        k=3,
    )

    assert result == 1.0


def test_hit_rate_at_k_returns_zero_without_a_hit() -> None:
    """Hit Rate must be zero when the top K has no relevant movie."""

    recommended_items = [10, 20, 30]
    relevant_items = {40, 50}

    result = hit_rate_at_k(
        recommended_items,
        relevant_items,
        k=3,
    )

    assert result == 0.0


def test_hit_rate_at_k_ignores_relevant_items_below_k() -> None:
    """A relevant movie below position K must not count as a hit."""

    recommended_items = [10, 20, 30, 40]
    relevant_items = {40}

    result = hit_rate_at_k(
        recommended_items,
        relevant_items,
        k=3,
    )

    assert result == 0.0


def test_hit_rate_at_k_returns_zero_without_relevant_items() -> None:
    """A case without relevant movies has zero Hit Rate."""

    result = hit_rate_at_k(
        recommended_items=[10, 20, 30],
        relevant_items=set(),
        k=3,
    )

    assert result == 0.0


@pytest.mark.parametrize("invalid_k", [0, -1])
def test_hit_rate_at_k_rejects_invalid_k(invalid_k: int) -> None:
    """K must be a positive integer."""

    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        hit_rate_at_k(
            recommended_items=[10, 20],
            relevant_items={10},
            k=invalid_k,
        )


def test_hit_rate_at_k_rejects_duplicate_recommendations() -> None:
    """A ranked recommendation list must not repeat a movie."""

    with pytest.raises(
        ValueError,
        match="must not contain duplicates",
    ):
        hit_rate_at_k(
            recommended_items=[10, 20, 10],
            relevant_items={10},
            k=3,
        )




def test_ndcg_at_k_matches_manual_calculation() -> None:
    """NDCG must match an example calculated by hand."""

    recommended_items = [10, 20, 30, 40]
    relevant_items = {10, 30}

    result = ndcg_at_k(
        recommended_items,
        relevant_items,
        k=3,
    )

    assert result == pytest.approx(0.9197207891)


def test_ndcg_at_k_returns_one_for_an_ideal_ranking() -> None:
    """NDCG must be one when relevant movies occupy the top ranks."""

    recommended_items = [10, 20, 30, 40]
    relevant_items = {10, 20}

    result = ndcg_at_k(
        recommended_items,
        relevant_items,
        k=3,
    )

    assert result == pytest.approx(1.0)


def test_ndcg_at_k_rewards_a_better_ranking() -> None:
    """Moving a relevant movie upward must improve NDCG."""

    relevant_items = {10}

    better_result = ndcg_at_k(
        recommended_items=[10, 20, 30],
        relevant_items=relevant_items,
        k=3,
    )
    worse_result = ndcg_at_k(
        recommended_items=[20, 30, 10],
        relevant_items=relevant_items,
        k=3,
    )

    assert better_result > worse_result


def test_ndcg_at_k_ignores_relevant_items_below_k() -> None:
    """A relevant movie below position K must not contribute."""

    recommended_items = [10, 20, 30, 40]
    relevant_items = {40}

    result = ndcg_at_k(
        recommended_items,
        relevant_items,
        k=3,
    )

    assert result == 0.0


def test_ndcg_at_k_returns_zero_without_relevant_items() -> None:
    """A case without relevant movies has zero NDCG."""

    result = ndcg_at_k(
        recommended_items=[10, 20, 30],
        relevant_items=set(),
        k=3,
    )

    assert result == 0.0


@pytest.mark.parametrize("invalid_k", [0, -1])
def test_ndcg_at_k_rejects_invalid_k(invalid_k: int) -> None:
    """K must be a positive integer."""

    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        ndcg_at_k(
            recommended_items=[10, 20],
            relevant_items={10},
            k=invalid_k,
        )


def test_ndcg_at_k_rejects_duplicate_recommendations() -> None:
    """A ranked recommendation list must not repeat a movie."""

    with pytest.raises(
        ValueError,
        match="must not contain duplicates",
    ):
        ndcg_at_k(
            recommended_items=[10, 20, 10],
            relevant_items={10},
            k=3,
        )




def test_coverage_at_k_matches_manual_calculation() -> None:
    """Coverage must match an example calculated by hand."""

    recommendation_lists = [
        [1, 2, 3],
        [2, 4, 5],
    ]
    catalog_items = {1, 2, 3, 4, 5}

    result = coverage_at_k(
        recommendation_lists,
        catalog_items,
        k=2,
    )

    assert result == pytest.approx(3 / 5)


def test_coverage_at_k_only_uses_top_k_items() -> None:
    """Movies below position K must not contribute to coverage."""

    recommendation_lists = [
        [1, 2, 3, 4],
    ]
    catalog_items = {1, 2, 3, 4, 5}

    result = coverage_at_k(
        recommendation_lists,
        catalog_items,
        k=2,
    )

    assert result == pytest.approx(2 / 5)


def test_coverage_at_k_returns_one_for_full_coverage() -> None:
    """Coverage must be one when the lists cover the full catalog."""

    recommendation_lists = [
        [1, 2],
        [3, 4],
    ]
    catalog_items = {1, 2, 3, 4}

    result = coverage_at_k(
        recommendation_lists,
        catalog_items,
        k=2,
    )

    assert result == 1.0


def test_coverage_at_k_returns_zero_without_recommendations() -> None:
    """No recommendation lists produce zero catalog coverage."""

    result = coverage_at_k(
        recommendation_lists=[],
        catalog_items={1, 2, 3},
        k=2,
    )

    assert result == 0.0


@pytest.mark.parametrize("invalid_k", [0, -1])
def test_coverage_at_k_rejects_invalid_k(invalid_k: int) -> None:
    """K must be a positive integer."""

    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        coverage_at_k(
            recommendation_lists=[[1, 2]],
            catalog_items={1, 2},
            k=invalid_k,
        )


def test_coverage_at_k_rejects_an_empty_catalog() -> None:
    """Coverage is undefined when the eligible catalog is empty."""

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        coverage_at_k(
            recommendation_lists=[],
            catalog_items=set(),
            k=10,
        )


def test_coverage_at_k_rejects_movies_outside_catalog() -> None:
    """Every recommended movie must belong to the catalog."""

    with pytest.raises(
        ValueError,
        match="must belong to the catalog",
    ):
        coverage_at_k(
            recommendation_lists=[[1, 2, 99]],
            catalog_items={1, 2, 3},
            k=3,
        )


def test_coverage_at_k_rejects_duplicate_items_in_one_list() -> None:
    """A single ranked list must not repeat a movie."""

    with pytest.raises(
        ValueError,
        match="must not contain duplicates",
    ):
        coverage_at_k(
            recommendation_lists=[[1, 2, 1]],
            catalog_items={1, 2, 3},
            k=3,
        )
