"""Aggregation strategies for group movie recommendation."""

import math
from collections.abc import Sequence
from typing import Final


MIN_SCORE: Final[float] = 1.0
MAX_SCORE: Final[float] = 5.0
MIN_GROUP_SIZE: Final[int] = 2
MAX_GROUP_SIZE: Final[int] = 5


def _validate_member_scores(
    member_scores: Sequence[float],
) -> tuple[float, ...]:
    """Validate and normalize predicted scores for one movie.

    CineMatch accepts groups containing two to five members. During
    week 1, predictions use the MovieLens rating scale from 1 to 5.
    """
    scores = tuple(float(score) for score in member_scores)

    if not MIN_GROUP_SIZE <= len(scores) <= MAX_GROUP_SIZE:
        raise ValueError(
            "Member scores must contain between 2 and 5 values"
        )

    if any(not math.isfinite(score) for score in scores):
        raise ValueError("Member scores must be finite")

    if any(
        score < MIN_SCORE or score > MAX_SCORE
        for score in scores
    ):
        raise ValueError(
            "Member scores must be between 1.0 and 5.0"
        )

    return scores


def average_score(
    member_scores: Sequence[float],
) -> float:
    """Return the arithmetic mean of all validated member scores."""
    scores = _validate_member_scores(member_scores)

    return math.fsum(scores) / len(scores)
