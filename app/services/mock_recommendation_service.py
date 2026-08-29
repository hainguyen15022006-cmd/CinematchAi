"""Dữ liệu recommendation cố định phục vụ tích hợp Frontend tuần 1."""

from app.schemas.run import (
    GroupRecommendationOut,
    RecommendationRequest,
)
from cinematch.recommendation.group import (
    DEFAULT_MISERY_THRESHOLD,
)
from cinematch.recommendation.response import (
    MovieResponseMetadata,
    build_group_recommendation_response,
    group_response_to_backend_payload,
)


MISERY_THRESHOLD = DEFAULT_MISERY_THRESHOLD
MOCK_USERS = (
    (1, "Thanh vien 1"),
    (2, "Thanh vien 2"),
    (3, "Thanh vien 3"),
)
MOCK_CANDIDATES = (
    (1, "Toy Story (1995)", ["Animation", "Children", "Comedy"], (4.8, 4.4, 4.6)),
    (2, "GoldenEye (1995)", ["Action", "Adventure", "Thriller"], (4.5, 3.9, 4.2)),
    (3, "Four Rooms (1995)", ["Thriller"], (3.6, 3.8, 3.5)),
    (4, "Get Shorty (1995)", ["Action", "Comedy", "Drama"], (4.2, 4.0, 4.1)),
    (5, "Copycat (1995)", ["Crime", "Drama", "Thriller"], (4.1, 1.5, 3.9)),
    (6, "Shanghai Triad (1995)", ["Drama"], (3.7, 3.5, 3.8)),
    (7, "Twelve Monkeys (1995)", ["Drama", "Sci-Fi"], (4.7, 4.1, 4.5)),
    (8, "Babe (1995)", ["Children", "Comedy", "Drama"], (4.3, 4.6, 4.0)),
    (9, "Dead Man Walking (1995)", ["Drama"], (4.0, 3.7, 4.4)),
    (10, "Richard III (1995)", ["Drama", "War"], (3.9, 3.6, 3.8)),
    (11, "Seven (1995)", ["Crime", "Thriller"], (4.6, 2.1, 4.3)),
    (12, "Usual Suspects, The (1995)", ["Crime", "Thriller"], (4.9, 4.3, 4.7)),
)


def build_mock_recommendations(
    request: RecommendationRequest,
) -> GroupRecommendationOut:
    """Tạo Top-K ổn định bằng đúng logic Group Recommendation lõi."""
    candidate_scores = {
        movie_id: scores
        for movie_id, _title, _genres, scores in MOCK_CANDIDATES
    }
    movie_metadata = {
        movie_id: MovieResponseMetadata(
            title=title,
            genres=tuple(genres),
        )
        for movie_id, title, genres, _scores in MOCK_CANDIDATES
    }
    member_user_ids = tuple(user_id for user_id, _name in MOCK_USERS)
    member_display_names = dict(MOCK_USERS)

    core_response = build_group_recommendation_response(
        room_id=request.room_id,
        member_user_ids=member_user_ids,
        candidate_scores=candidate_scores,
        strategy=request.strategy.value,
        top_k=request.top_k,
        misery_threshold=MISERY_THRESHOLD,
        member_display_names=member_display_names,
    )
    payload = group_response_to_backend_payload(
        core_response,
        movie_metadata,
    )

    return GroupRecommendationOut.model_validate(payload)
