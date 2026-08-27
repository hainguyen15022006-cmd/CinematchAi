from enum import Enum
from datetime import datetime
from typing import List, Optional

from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field

from app.schemas.movie import MovieOut
from app.schemas.user import UserOut


class AggregationStrategy(str, Enum):
    """Các chiến lược tổng hợp điểm được CineMatch hỗ trợ."""

    AVERAGE = "average"
    LEAST_MISERY = "least_misery"
    AVERAGE_WITHOUT_MISERY = "average_without_misery"


class RecommendationRequest(BaseModel):
    """Request của Frontend khi chủ phòng yêu cầu Top-K đề xuất."""

    room_id: int = Field(gt=0)
    strategy: AggregationStrategy
    top_k: int = Field(default=10, ge=1, le=20)


class MemberScore(BaseModel):
    """Điểm AI dự đoán cho một thành viên đối với một phim."""

    user_id: int = Field(gt=0)
    display_name: Optional[str] = None
    predicted_score: float = Field(ge=1, le=5)


class RunItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: int
    movie_id: int = Field(
        validation_alias=AliasChoices(
            AliasPath("movie", "movielens_id"),
            "movie_id",
        )
    )
    rank: int
    ai_score: Optional[float] = None
    movie: Optional[MovieOut] = None

class RecommendationRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    room_id: int
    status: str
    created_at: datetime
    items: List[RunItemOut] = Field(default_factory=list)

class VoteCreate(BaseModel):
    movie_id: int = Field(
        gt=0,
        description="ID gốc MovieLens",
    )
    vote_value: float = Field(default=1, ge=-1, le=1)


class VoteOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: int
    user_id: int
    movie_id: int = Field(
        validation_alias=AliasChoices(
            AliasPath("movie", "movielens_id"),
            "movie_id",
        )
    )
    vote_value: float
    created_at: datetime
    user: Optional[UserOut] = None

class RunResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: int = Field(validation_alias=AliasChoices("run_id", "id"))
    winner_movie: Optional[MovieOut] = None
    group_score: Optional[float] = None
    disagreement: Optional[float] = None
    votes: List[VoteOut] = Field(default_factory=list)

class RecommendedMovie(BaseModel):
    """Một phim trong Top-K cùng dữ liệu fairness và giải thích."""

    movie_id: int = Field(gt=0, description="ID gốc MovieLens")
    rank: int = Field(ge=1)
    title: str = Field(min_length=1)
    genres: List[str] = Field(default_factory=list)
    poster_url: Optional[str] = None
    runtime_minutes: Optional[int] = Field(default=None, gt=0)
    group_score: float = Field(
        ge=1,
        le=5,
        validation_alias=AliasChoices("group_score", "score"),
    )
    minimum_score: float = Field(ge=1, le=5)
    disagreement: float = Field(ge=0)
    member_scores: List[MemberScore] = Field(default_factory=list)
    misery_warning: bool
    explanations: List[str] = Field(default_factory=list)


class GroupRecommendationOut(BaseModel):
    """Response Top-K dùng chung cho Backend và Frontend.

    ``top_movies`` từ bản nháp cũ vẫn được chấp nhận khi validate để code
    thử nghiệm của Chúc không bị gãy. Khi serialize, tên chuẩn được dùng là
    ``recommendations``.
    """

    schema_version: str = "1.0"
    room_id: int = Field(gt=0)
    strategy: AggregationStrategy
    recommendations: List[RecommendedMovie] = Field(
        default_factory=list,
        max_length=20,
        validation_alias=AliasChoices("recommendations", "top_movies"),
    )
