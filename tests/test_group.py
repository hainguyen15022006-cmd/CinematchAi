"""Tests for group recommendation aggregation strategies."""

import pytest

from cinematch.recommendation.group import average_score


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
