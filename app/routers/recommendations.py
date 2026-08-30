"""Mock recommendation endpoint for week-one Frontend integration."""

from fastapi import APIRouter

from app.schemas.run import (
    GroupRecommendationOut,
    RecommendationRequest,
)
from app.services.mock_recommendation_service import (
    build_mock_recommendations,
)


router = APIRouter(prefix="/recommend", tags=["Mock Recommendation"])


@router.post(
    "/mock",
    response_model=GroupRecommendationOut,
    summary="Build a mock Top-K following the Group Recommendation v1 schema",
)
def recommend_mock(
    request: RecommendationRequest,
) -> GroupRecommendationOut:
    """Return fixed data; does not call the AI and does not write to the database."""
    return build_mock_recommendations(request)
