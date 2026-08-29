"""Dữ liệu recommendation cố định phục vụ tích hợp Frontend tuần 1."""

from statistics import fmean, pstdev

from app.schemas.run import (
    AggregationStrategy,
    GroupRecommendationOut,
    MemberScore,
    RecommendationRequest,
    RecommendedMovie,
)
from cinematch.recommendation.group import (
    DEFAULT_MISERY_THRESHOLD,
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


def _group_score(
    scores: tuple[float, ...],
    strategy: AggregationStrategy,
) -> float:
    if strategy is AggregationStrategy.LEAST_MISERY:
        return min(scores)
    return fmean(scores)


def build_mock_recommendations(
    request: RecommendationRequest,
) -> GroupRecommendationOut:
    """Tạo Top-K ổn định để FE test mà không cần model hoặc database."""
    candidates: list[RecommendedMovie] = []

    for movie_id, title, genres, scores in MOCK_CANDIDATES:
        minimum_score = min(scores)
        misery_warning = minimum_score < MISERY_THRESHOLD
        if (
            request.strategy
            is AggregationStrategy.AVERAGE_WITHOUT_MISERY
            and misery_warning
        ):
            continue

        member_scores = [
            MemberScore(
                user_id=user_id,
                display_name=display_name,
                predicted_score=score,
            )
            for (user_id, display_name), score in zip(MOCK_USERS, scores)
        ]
        explanations = [
            f"Phu hop the loai chung: {', '.join(genres)}",
            f"Tong hop bang chien luoc: {request.strategy.value}",
        ]
        if misery_warning:
            explanations.append(
                f"Co thanh vien duoi nguong misery {MISERY_THRESHOLD:.1f}"
            )
        else:
            explanations.append("Khong co thanh vien bi diem qua thap")

        candidates.append(
            RecommendedMovie(
                movie_id=movie_id,
                rank=1,
                title=title,
                genres=genres,
                poster_url=None,
                runtime_minutes=None,
                group_score=round(
                    _group_score(scores, request.strategy), 4
                ),
                minimum_score=minimum_score,
                disagreement=round(pstdev(scores), 4),
                member_scores=member_scores,
                misery_warning=misery_warning,
                explanations=explanations,
            )
        )

    candidates.sort(key=lambda item: (-item.group_score, item.movie_id))
    top_items = candidates[: request.top_k]
    ranked_items = [
        item.model_copy(update={"rank": rank})
        for rank, item in enumerate(top_items, start=1)
    ]

    return GroupRecommendationOut(
        room_id=request.room_id,
        strategy=request.strategy,
        recommendations=ranked_items,
    )
