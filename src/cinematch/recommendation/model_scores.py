"""Adapters between individual model outputs and group scoring."""

import math
from collections.abc import Sequence
from numbers import Integral, Real

from cinematch.recommendation.group import (
    MAX_GROUP_SIZE,
    MAX_SCORE,
    MIN_GROUP_SIZE,
    MIN_SCORE,
)


def prepare_group_candidate_scores(
    movie_ids: Sequence[int],
    member_score_rows: Sequence[Sequence[float]],
) -> dict[int, tuple[float, ...]]:
    """Align and normalize individual predictions by movie.

    Each input row contains one member's predictions across every
    movie. The output maps each original MovieLens movie ID to the
    scores of all members in the same member order.

    Finite model outputs are clamped to the MovieLens rating scale
    from 1.0 to 5.0 before entering group aggregation.

    Args:
        movie_ids:
            Original MovieLens movie IDs ordered like model outputs.
        member_score_rows:
            One score row per group member. Every row must contain
            one prediction for every movie ID.

    Returns:
        Candidate scores indexed by original MovieLens movie ID.

    Raises:
        ValueError:
            If movie IDs, group size, row lengths or scores violate
            the shared Model-to-Group contract.
    """

    normalized_movie_ids = tuple(movie_ids)

    if not normalized_movie_ids:
        raise ValueError("Movie IDs must not be empty")

    if any(
        isinstance(movie_id, bool)
        or not isinstance(movie_id, Integral)
        or int(movie_id) <= 0
        for movie_id in normalized_movie_ids
    ):
        raise ValueError(
            "Movie IDs must contain positive integers"
        )

    normalized_movie_ids = tuple(
        int(movie_id)
        for movie_id in normalized_movie_ids
    )

    if len(set(normalized_movie_ids)) != len(
        normalized_movie_ids
    ):
        raise ValueError("Movie IDs must not contain duplicates")

    score_rows = tuple(
        tuple(row)
        for row in member_score_rows
    )

    if not MIN_GROUP_SIZE <= len(score_rows) <= MAX_GROUP_SIZE:
        raise ValueError(
            "Model output must contain between 2 and 5 member rows"
        )

    if any(
        len(row) != len(normalized_movie_ids)
        for row in score_rows
    ):
        raise ValueError(
            "Every member score row must match the movie count"
        )

    candidate_scores: dict[int, tuple[float, ...]] = {}

    for movie_position, movie_id in enumerate(
        normalized_movie_ids
    ):
        normalized_scores: list[float] = []

        for row in score_rows:
            raw_score = row[movie_position]

            if (
                isinstance(raw_score, bool)
                or not isinstance(raw_score, Real)
                or not math.isfinite(float(raw_score))
            ):
                raise ValueError(
                    "Model scores must contain finite numbers"
                )

            score = float(raw_score)
            clamped_score = min(
                MAX_SCORE,
                max(MIN_SCORE, score),
            )
            normalized_scores.append(clamped_score)

        candidate_scores[movie_id] = tuple(
            normalized_scores
        )

    return candidate_scores