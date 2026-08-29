"""Tests for the Model-to-Group score adapter."""

import pytest

from cinematch.recommendation import (
    prepare_group_candidate_scores,
)


def test_prepare_group_candidate_scores_aligns_by_movie() -> None:
    """Member score rows must be transposed into movie scores."""

    result = prepare_group_candidate_scores(
        movie_ids=[50, 100, 200],
        member_score_rows=[
            [4.8, 3.5, 4.0],
            [4.4, 4.1, 3.0],
        ],
    )

    assert result == {
        50: (4.8, 4.4),
        100: (3.5, 4.1),
        200: (4.0, 3.0),
    }


def test_prepare_group_candidate_scores_clamps_rating_range() -> None:
    """Finite neural outputs must be clamped to the 1-5 contract."""

    result = prepare_group_candidate_scores(
        movie_ids=[50, 100],
        member_score_rows=[
            [-0.04, 4.5],
            [0.16, 6.0],
        ],
    )

    assert result == {
        50: (1.0, 1.0),
        100: (4.5, 5.0),
    }


@pytest.mark.parametrize(
    "invalid_score",
    [
        float("nan"),
        float("inf"),
    ],
)
def test_prepare_group_candidate_scores_rejects_non_finite(
    invalid_score: float,
) -> None:
    """NaN and infinite model outputs must never enter ranking."""

    with pytest.raises(
        ValueError,
        match="finite numbers",
    ):
        prepare_group_candidate_scores(
            movie_ids=[50],
            member_score_rows=[
                [4.0],
                [invalid_score],
            ],
        )


def test_prepare_group_candidate_scores_rejects_wrong_row_length() -> None:
    """Every member must have one prediction for every movie."""

    with pytest.raises(
        ValueError,
        match="must match the movie count",
    ):
        prepare_group_candidate_scores(
            movie_ids=[50, 100],
            member_score_rows=[
                [4.0, 4.5],
                [3.5],
            ],
        )


def test_prepare_group_candidate_scores_rejects_duplicate_movies() -> None:
    """Decoded MovieLens movie IDs must remain unique."""

    with pytest.raises(
        ValueError,
        match="must not contain duplicates",
    ):
        prepare_group_candidate_scores(
            movie_ids=[50, 50],
            member_score_rows=[
                [4.0, 4.5],
                [3.5, 4.0],
            ],
        )


def test_prepare_group_candidate_scores_rejects_one_member() -> None:
    """CineMatch group recommendation requires at least two members."""

    with pytest.raises(
        ValueError,
        match="between 2 and 5",
    ):
        prepare_group_candidate_scores(
            movie_ids=[50],
            member_score_rows=[
                [4.0],
            ],
        )