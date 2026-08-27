"""Endpoint recommendation giả phục vụ tích hợp Frontend tuần 1."""

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
    summary="Tạo Top-K giả theo schema Group Recommendation v1",
)
def recommend_mock(
    request: RecommendationRequest,
) -> GroupRecommendationOut:
    """Trả dữ liệu cố định; không gọi AI và không ghi database."""
    return build_mock_recommendations(request)
