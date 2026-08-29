"""Tests for group recommendation aggregation strategies."""
import math
import pytest

from cinematch.recommendation.group import (
    AggregationStrategy,
    GroupItemScore,
    aggregate_item_scores,
    average_score,
    average_without_misery_score,
    disagreement_score,
    least_misery_score,
    rank_group_candidates,
)

def test_average_score_matches_manual_calculation() -> None:
    """Average must match the example calculated by hand."""
    scores = [5.0, 5.0, 5.0, 1.5]

    result = average_score(scores)

    assert result == pytest.approx(4.125)


@pytest.mark.parametrize(
    "scores",
    [
        [],
        [4.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 5.0],
    ],
)
def test_average_score_rejects_invalid_group_size(
    scores: list[float],
) -> None:
    """Only groups containing two to five members are valid."""
    with pytest.raises(
        ValueError,
        match="between 2 and 5",
    ):
        average_score(scores)


@pytest.mark.parametrize(
    "scores",
    [
        [4.0, float("nan")],
        [4.0, float("inf")],
    ],
)
def test_average_score_rejects_non_finite_values(
    scores: list[float],
) -> None:
    """NaN and infinite predictions must not enter ranking."""
    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        average_score(scores)


@pytest.mark.parametrize(
    "scores",
    [
        [0.5, 4.0],
        [4.0, 5.5],
    ],
)
def test_average_score_rejects_scores_outside_range(
    scores: list[float],
) -> None:
    """Week 1 predictions must remain on the 1-5 scale."""
    with pytest.raises(
        ValueError,
        match="between 1.0 and 5.0",
    ):
        average_score(scores)


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([5.0, 5.0, 5.0, 1.5], 1.5),
        ([4.0, 4.0, 4.0, 4.0], 4.0),
        ([4.5, 3.5, 4.0, 3.0], 3.0),
    ],
)
def test_least_misery_matches_manual_calculation(
    scores: list[float],
    expected: float,
) -> None:
    """Least Misery must return the lowest member score."""
    result = least_misery_score(scores)

    assert result == pytest.approx(expected)


def test_least_misery_reuses_score_validation() -> None:
    """Least Misery must reject invalid model predictions."""
    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        least_misery_score([4.0, float("nan")])

@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([5.0, 5.0, 5.0, 1.5], None),
        ([4.0, 4.0, 4.0, 4.0], 4.0),
        ([4.5, 3.5, 4.0, 3.0], 3.75),
    ],
)
def test_average_without_misery_matches_manual_calculation(
    scores: list[float],
    expected: float | None,
) -> None:
    """Movies below threshold are rejected; others use Average."""
    result = average_without_misery_score(
        scores,
        misery_threshold=2.0,
    )

    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_average_without_misery_keeps_threshold_equality() -> None:
    """A score equal to the threshold must remain eligible."""
    result = average_without_misery_score(
        [2.0, 4.0],
        misery_threshold=2.0,
    )

    assert result == pytest.approx(3.0)


@pytest.mark.parametrize(
    "threshold",
    [
        float("nan"),
        0.5,
        5.5,
    ],
)
def test_average_without_misery_rejects_invalid_threshold(
    threshold: float,
) -> None:
    """Threshold must be finite and use the same 1-5 scale."""
    expected_message = (
        "must be finite"
        if not math.isfinite(threshold)
        else "between 1.0 and 5.0"
    )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        average_without_misery_score(
            [3.0, 4.0],
            misery_threshold=threshold,
        )

@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([5.0, 5.0, 5.0, 1.5], 1.5155444566),
        ([4.0, 4.0, 4.0, 4.0], 0.0),
        ([4.5, 3.5, 4.0, 3.0], 0.5590169944),
    ],
)
def test_disagreement_matches_manual_calculation(
    scores: list[float],
    expected: float,
) -> None:
    """Disagreement must match population standard deviation."""
    result = disagreement_score(scores)

    assert result == pytest.approx(expected)


def test_disagreement_reuses_score_validation() -> None:
    """Disagreement must reject scores outside the contract."""
    with pytest.raises(
        ValueError,
        match="between 1.0 and 5.0",
    ):
        disagreement_score([4.0, 5.5])

@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (AggregationStrategy.AVERAGE, 4.125),
        ("least_misery", 1.5),
    ],
)
def test_aggregate_item_scores_selects_strategy(
    strategy: AggregationStrategy | str,
    expected: float,
) -> None:
    """The selected strategy must determine the group score."""
    result = aggregate_item_scores(
        movie_id=10,
        member_scores=[5.0, 5.0, 5.0, 1.5],
        strategy=strategy,
    )

    assert isinstance(result, GroupItemScore)
    assert result.group_score == pytest.approx(expected)
    assert result.minimum_score == pytest.approx(1.5)
    assert result.member_scores == (5.0, 5.0, 5.0, 1.5)


def test_aggregate_item_scores_returns_explanation_metrics() -> None:
    """An eligible movie must contain all group explanation data."""
    result = aggregate_item_scores(
        movie_id=20,
        member_scores=[4.0, 4.0, 4.0, 4.0],
        strategy="average_without_misery",
        misery_threshold=2.0,
    )

    assert isinstance(result, GroupItemScore)
    assert result.movie_id == 20
    assert result.group_score == pytest.approx(4.0)
    assert result.minimum_score == pytest.approx(4.0)
    assert result.disagreement == pytest.approx(0.0)
    assert result.member_scores == (4.0, 4.0, 4.0, 4.0)


def test_aggregate_item_scores_returns_none_for_misery() -> None:
    """AWM must reject a movie below the misery threshold."""
    result = aggregate_item_scores(
        movie_id=30,
        member_scores=[5.0, 5.0, 5.0, 1.5],
        strategy=AggregationStrategy.AVERAGE_WITHOUT_MISERY,
        misery_threshold=2.0,
    )

    assert result is None


def test_aggregate_item_scores_rejects_unknown_strategy() -> None:
    """Misspelled strategy names must fail clearly."""
    with pytest.raises(
        ValueError,
        match="Unknown aggregation strategy",
    ):
        aggregate_item_scores(
            movie_id=40,
            member_scores=[3.0, 4.0],
            strategy="avarage",
        )


@pytest.mark.parametrize("movie_id", [0, -1])
def test_aggregate_item_scores_rejects_invalid_movie_id(
    movie_id: int,
) -> None:
    """Movie IDs must be positive integers."""
    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        aggregate_item_scores(
            movie_id=movie_id,
            member_scores=[3.0, 4.0],
            strategy="average",
        )

def test_rank_group_candidates_orders_by_group_score() -> None:
    """Candidates with higher Average must appear first."""
    candidates = {
        101: [5.0, 5.0, 5.0, 1.5],
        102: [4.0, 4.0, 4.0, 4.0],
        103: [4.5, 3.5, 4.0, 3.0],
    }

    result = rank_group_candidates(
        candidates,
        strategy="average",
    )

    assert [item.movie_id for item in result] == [
        101,
        102,
        103,
    ]


def test_rank_group_candidates_removes_misery_items() -> None:
    """AWM must omit candidates below the threshold."""
    candidates = {
        101: [5.0, 5.0, 5.0, 1.5],
        102: [4.0, 4.0, 4.0, 4.0],
        103: [4.5, 3.5, 4.0, 3.0],
    }

    result = rank_group_candidates(
        candidates,
        strategy="average_without_misery",
        misery_threshold=2.0,
    )

    assert [item.movie_id for item in result] == [102, 103]


def test_ranking_prefers_higher_minimum_score_on_tie() -> None:
    """Minimum score is the first tie-break criterion."""
    candidates = {
        10: [5.0, 3.0],
        20: [4.0, 4.0],
    }

    result = rank_group_candidates(
        candidates,
        strategy="average",
    )

    assert [item.movie_id for item in result] == [20, 10]


def test_ranking_prefers_lower_disagreement_on_tie() -> None:
    """Disagreement is used after group and minimum scores."""
    candidates = {
        10: [3.0, 4.0, 5.0],
        20: [3.0, 4.5, 4.5],
    }

    result = rank_group_candidates(
        candidates,
        strategy="average",
    )

    assert [item.movie_id for item in result] == [20, 10]


def test_ranking_uses_movie_id_as_final_tie_break() -> None:
    """Movie ID makes fully tied rankings deterministic."""
    candidates = {
        40: [5.0, 3.0],
        30: [3.0, 5.0],
    }

    result = rank_group_candidates(
        candidates,
        strategy="average",
    )

    assert [item.movie_id for item in result] == [30, 40]


def test_rank_group_candidates_limits_top_k() -> None:
    """Only the requested number of candidates is returned."""
    candidates = {
        101: [5.0, 5.0],
        102: [4.0, 4.0],
        103: [3.0, 3.0],
    }

    result = rank_group_candidates(
        candidates,
        strategy="average",
        top_k=2,
    )

    assert len(result) == 2
    assert [item.movie_id for item in result] == [101, 102]


def test_rank_group_candidates_rejects_empty_input() -> None:
    """Ranking requires at least one candidate."""
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        rank_group_candidates(
            {},
            strategy="average",
        )


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_rank_group_candidates_rejects_invalid_top_k(
    top_k: int,
) -> None:
    """Top K must be a positive integer."""
    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        rank_group_candidates(
            {101: [4.0, 4.0]},
            strategy="average",
            top_k=top_k,
        )


def test_ranking_requires_same_members_for_every_movie() -> None:
    """Every candidate must have scores for the same group."""
    candidates = {
        101: [4.0, 4.0],
        102: [4.0, 4.0, 4.0],
    }

    with pytest.raises(
        ValueError,
        match="same number of members",
    ):
        rank_group_candidates(
            candidates,
            strategy="average",
        )