"""Aggregation strategies for group movie recommendation."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final


MIN_SCORE: Final[float] = 1.0
MAX_SCORE: Final[float] = 5.0
MIN_GROUP_SIZE: Final[int] = 2
MAX_GROUP_SIZE: Final[int] = 5
DEFAULT_MISERY_THRESHOLD: Final[float] = 2.0

class AggregationStrategy(str, Enum):
    """Supported group aggregation strategies."""

    AVERAGE = "average"
    LEAST_MISERY = "least_misery"
    AVERAGE_WITHOUT_MISERY = "average_without_misery"

@dataclass(frozen=True)
class GroupItemScore:
    """Aggregated group result for one movie."""

    movie_id: int
    group_score: float
    minimum_score: float
    disagreement: float
    member_scores: tuple[float, ...]

def _validate_member_scores(
    member_scores: Sequence[float],
) -> tuple[float, ...]:
    """Validate and normalize predicted member scores.

    CineMatch accepts groups containing two to five members.
    During week 1, every predicted score must use the
    MovieLens rating scale from 1 to 5.

    Args:
        member_scores:
            Predicted scores for one movie, ordered by member.

    Returns:
        The validated scores as an immutable tuple.

    Raises:
        ValueError:
            If the group size is invalid, a score is not finite,
            or a score is outside the supported range.
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
    """Calculate the arithmetic mean of all member scores.

    Every member has equal influence. A high score from several
    members can compensate for a low score from another member.

    Args:
        member_scores:
            Predicted scores for one movie.

    Returns:
        The arithmetic mean of the validated scores.
    """
    scores = _validate_member_scores(member_scores)

    return math.fsum(scores) / len(scores)

def least_misery_score(
    member_scores: Sequence[float],
) -> float:
    """Return the lowest validated member score.

    Least Misery protects the least satisfied member by using
    their score as the score of the entire group.
    """
    scores = _validate_member_scores(member_scores)

    return min(scores)

def average_without_misery_score(
    member_scores: Sequence[float],
    misery_threshold: float = DEFAULT_MISERY_THRESHOLD,
) -> float | None:
    """Return Average after rejecting movies below the threshold.

    A return value of None means that the movie is not eligible
    because at least one member score is below the misery threshold.
    """
    scores = _validate_member_scores(member_scores)
    threshold = float(misery_threshold)

    if not math.isfinite(threshold):
        raise ValueError("Misery threshold must be finite")

    if threshold < MIN_SCORE or threshold > MAX_SCORE:
        raise ValueError(
            "Misery threshold must be between 1.0 and 5.0"
        )

    if min(scores) < threshold:
        return None

    return math.fsum(scores) / len(scores)

def disagreement_score(
    member_scores: Sequence[float],
) -> float:
    """Return population standard deviation of member scores.

    A lower value means that members agree more closely. A value
    of zero means all members have exactly the same score.
    """
    scores = _validate_member_scores(member_scores)
    mean_score = math.fsum(scores) / len(scores)

    squared_differences = [
        (score - mean_score) ** 2
        for score in scores
    ]
    population_variance = (
        math.fsum(squared_differences) / len(scores)
    )

    return math.sqrt(population_variance)

def aggregate_item_scores(
    movie_id: int,
    member_scores: Sequence[float],
    strategy: AggregationStrategy | str,
    misery_threshold: float = DEFAULT_MISERY_THRESHOLD,
) -> GroupItemScore | None:
    """Aggregate member predictions for one movie.

    None is returned only when Average Without Misery rejects
    the movie for falling below the configured threshold.
    """
    if (
        isinstance(movie_id, bool)
        or not isinstance(movie_id, int)
        or movie_id <= 0
    ):
        raise ValueError("Movie ID must be a positive integer")

    scores = _validate_member_scores(member_scores)

    try:
        selected_strategy = AggregationStrategy(strategy)
    except ValueError as error:
        raise ValueError(
            f"Unknown aggregation strategy: {strategy}"
        ) from error

    if selected_strategy is AggregationStrategy.AVERAGE:
        group_score = average_score(scores)
    elif selected_strategy is AggregationStrategy.LEAST_MISERY:
        group_score = least_misery_score(scores)
    else:
        group_score = average_without_misery_score(
            scores,
            misery_threshold=misery_threshold,
        )

        if group_score is None:
            return None

    return GroupItemScore(
        movie_id=movie_id,
        group_score=group_score,
        minimum_score=min(scores),
        disagreement=disagreement_score(scores),
        member_scores=scores,
    )

def rank_group_candidates(
    candidate_scores: Mapping[int, Sequence[float]],
    strategy: AggregationStrategy | str,
    misery_threshold: float = DEFAULT_MISERY_THRESHOLD,
    top_k: int = 10,
) -> list[GroupItemScore]:
    """Aggregate, sort and return the best group candidates.

    Ranking order:
    1. Higher group score.
    2. Higher minimum score.
    3. Lower disagreement.
    4. Lower movie ID for deterministic output.
    """
    if not candidate_scores:
        raise ValueError("Candidate scores cannot be empty")

    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k <= 0
    ):
        raise ValueError("Top K must be a positive integer")

    ranked_items: list[GroupItemScore] = []
    expected_member_count: int | None = None

    for movie_id, raw_scores in candidate_scores.items():
        scores = tuple(raw_scores)

        if expected_member_count is None:
            expected_member_count = len(scores)
        elif len(scores) != expected_member_count:
            raise ValueError(
                "All candidates must contain scores for the "
                "same number of members"
            )

        result = aggregate_item_scores(
            movie_id=movie_id,
            member_scores=scores,
            strategy=strategy,
            misery_threshold=misery_threshold,
        )

        if result is not None:
            ranked_items.append(result)

    ranked_items.sort(
        key=lambda item: (
            -item.group_score,
            -item.minimum_score,
            item.disagreement,
            item.movie_id,
        )
    )

    return ranked_items[:top_k]