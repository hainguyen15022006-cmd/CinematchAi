"""Public API for CineMatch group recommendation."""

from cinematch.recommendation.group import (
    DEFAULT_MISERY_THRESHOLD,
    AggregationStrategy,
    GroupItemScore,
    aggregate_item_scores,
    rank_group_candidates,
)
from cinematch.recommendation.model_scores import (
    prepare_group_candidate_scores,
)
from cinematch.recommendation.response import (
    GroupRecommendationItem,
    GroupRecommendationResponse,
    MemberPredictedScore,
    MovieResponseMetadata,
    build_group_recommendation_response,
    group_response_to_backend_payload,
)


__all__ = [
    "DEFAULT_MISERY_THRESHOLD",
    "AggregationStrategy",
    "GroupItemScore",
    "GroupRecommendationItem",
    "GroupRecommendationResponse",
    "MemberPredictedScore",
    "MovieResponseMetadata",
    "aggregate_item_scores",
    "build_group_recommendation_response",
    "group_response_to_backend_payload",
    "prepare_group_candidate_scores",
    "rank_group_candidates",
]