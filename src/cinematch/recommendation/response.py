"""Response models for group movie recommendations."""


from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cinematch.recommendation.group import (
    DEFAULT_MISERY_THRESHOLD,
    AggregationStrategy,
    rank_group_candidates,
)


@dataclass(frozen=True)
class MemberPredictedScore:
    """One model prediction for one group member and movie."""

    user_id: int
    predicted_score: float
    display_name: str | None = None



@dataclass(frozen=True)
class MovieResponseMetadata:
    """Movie metadata supplied by Backend for response serialization."""

    title: str
    genres: tuple[str, ...] = ()
    poster_url: str | None = None
    runtime_minutes: int | None = None


@dataclass(frozen=True)
class GroupRecommendationItem:
    """One ranked movie with group fairness information."""

    movie_id: int
    rank: int
    group_score: float
    minimum_score: float
    disagreement: float
    member_scores: tuple[MemberPredictedScore, ...]
    misery_warning: bool
    explanations: tuple[str, ...]


@dataclass(frozen=True)
class GroupRecommendationResponse:
    """Top-K group recommendation result passed to Backend."""

    room_id: int
    strategy: AggregationStrategy
    recommendations: tuple[GroupRecommendationItem, ...]
    schema_version: str = "1.0"



def build_group_recommendation_response(
    room_id: int,
    member_user_ids: Sequence[int],
    candidate_scores: Mapping[int, Sequence[float]],
    strategy: AggregationStrategy | str,
    *,
    top_k: int = 10,
    misery_threshold: float = DEFAULT_MISERY_THRESHOLD,
    member_display_names: Mapping[int, str] | None = None,
) -> GroupRecommendationResponse:
    """Build a ranked group response from member predictions.

    Args:
        room_id:
            Positive Backend room identifier.
        member_user_ids:
            User IDs ordered exactly like every member-score sequence.
        candidate_scores:
            Mapping from original MovieLens movie ID to predicted
            scores ordered by member.
        strategy:
            Group aggregation strategy.
        top_k:
            Maximum number of recommendations returned.
        misery_threshold:
            Score below which a misery warning is created.
        member_display_names:
            Optional display names indexed by user ID.

    Returns:
        A deterministic Top-K group recommendation response.

    Raises:
        ValueError:
            If room or member IDs are invalid, members are duplicated,
            score counts do not match member count, or group ranking
            inputs are invalid.
    """

    if (
        isinstance(room_id, bool)
        or not isinstance(room_id, int)
        or room_id <= 0
    ):
        raise ValueError("Room ID must be a positive integer")

    user_ids = tuple(member_user_ids)

    if not 2 <= len(user_ids) <= 5:
        raise ValueError(
            "Group response requires between 2 and 5 members"
        )

    if any(
        isinstance(user_id, bool)
        or not isinstance(user_id, int)
        or user_id <= 0
        for user_id in user_ids
    ):
        raise ValueError(
            "Member user IDs must be positive integers"
        )

    if len(set(user_ids)) != len(user_ids):
        raise ValueError("Member user IDs must be unique")

    for scores in candidate_scores.values():
        if len(scores) != len(user_ids):
            raise ValueError(
                "Every candidate score sequence must match "
                "the member count"
            )

    try:
        selected_strategy = AggregationStrategy(strategy)
    except ValueError as error:
        raise ValueError(
            f"Unknown aggregation strategy: {strategy}"
        ) from error

    ranked_scores = rank_group_candidates(
        candidate_scores=candidate_scores,
        strategy=selected_strategy,
        misery_threshold=misery_threshold,
        top_k=top_k,
    )

    display_names = member_display_names or {}
    recommendations: list[GroupRecommendationItem] = []

    for rank, item in enumerate(ranked_scores, start=1):
        member_scores = tuple(
            MemberPredictedScore(
                user_id=user_id,
                display_name=display_names.get(user_id),
                predicted_score=predicted_score,
            )
            for user_id, predicted_score in zip(
                user_ids,
                item.member_scores,
                strict=True,
            )
        )

        misery_warning = (
            item.minimum_score < misery_threshold
        )

        score_explanation = (
            f"Group score {item.group_score:.4f}; "
            f"minimum {item.minimum_score:.4f}; "
            f"disagreement {item.disagreement:.4f}"
        )

        if misery_warning:
            warning_explanation = (
                "At least one member is below misery threshold "
                f"{misery_threshold:.1f}"
            )
        else:
            warning_explanation = (
                "No member is below misery threshold "
                f"{misery_threshold:.1f}"
            )

        recommendations.append(
            GroupRecommendationItem(
                movie_id=item.movie_id,
                rank=rank,
                group_score=item.group_score,
                minimum_score=item.minimum_score,
                disagreement=item.disagreement,
                member_scores=member_scores,
                misery_warning=misery_warning,
                explanations=(
                    f"Aggregated using {selected_strategy.value}",
                    score_explanation,
                    warning_explanation,
                ),
            )
        )

    return GroupRecommendationResponse(
        room_id=room_id,
        strategy=selected_strategy,
        recommendations=tuple(recommendations),
    )



def group_response_to_backend_payload(
    response: GroupRecommendationResponse,
    movie_metadata: Mapping[int, MovieResponseMetadata],
) -> dict[str, object]:
    """Serialize a core group response using the Backend contract.

    Backend supplies display metadata indexed by original MovieLens
    movie ID. Group Recommendation supplies scores, fairness fields
    and explanations.

    Args:
        response:
            Ranked core response produced by Group Recommendation.
        movie_metadata:
            Backend movie metadata indexed by MovieLens movie ID.

    Returns:
        A JSON-compatible dictionary matching GroupRecommendationOut.

    Raises:
        ValueError:
            If metadata is missing or contains invalid display fields.
    """

    serialized_recommendations: list[dict[str, object]] = []

    for item in response.recommendations:
        metadata = movie_metadata.get(item.movie_id)

        if metadata is None:
            raise ValueError(
                f"Missing metadata for movie ID {item.movie_id}"
            )

        if not metadata.title.strip():
            raise ValueError("Movie title must not be empty")

        if any(
            not isinstance(genre, str) or not genre.strip()
            for genre in metadata.genres
        ):
            raise ValueError(
                "Movie genres must contain non-empty strings"
            )

        if (
            metadata.runtime_minutes is not None
            and (
                isinstance(metadata.runtime_minutes, bool)
                or not isinstance(metadata.runtime_minutes, int)
                or metadata.runtime_minutes <= 0
            )
        ):
            raise ValueError(
                "Runtime minutes must be a positive integer or None"
            )

        serialized_member_scores = [
            {
                "user_id": member_score.user_id,
                "display_name": member_score.display_name,
                "predicted_score": member_score.predicted_score,
            }
            for member_score in item.member_scores
        ]

        serialized_recommendations.append(
            {
                "movie_id": item.movie_id,
                "rank": item.rank,
                "title": metadata.title,
                "genres": list(metadata.genres),
                "poster_url": metadata.poster_url,
                "runtime_minutes": metadata.runtime_minutes,
                "group_score": item.group_score,
                "minimum_score": item.minimum_score,
                "disagreement": item.disagreement,
                "member_scores": serialized_member_scores,
                "misery_warning": item.misery_warning,
                "explanations": list(item.explanations),
            }
        )

    return {
        "schema_version": response.schema_version,
        "room_id": response.room_id,
        "strategy": response.strategy.value,
        "recommendations": serialized_recommendations,
    }