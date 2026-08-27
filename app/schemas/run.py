from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from app.schemas.movie import MovieOut
from app.schemas.user import UserOut

class RecommendedMovie(BaseModel):
    movie_id: int
    title: str
    score: float

class RunItemOut(BaseModel):
    id: int
    run_id: int
    movie_id: int
    rank: int
    ai_score: Optional[float] = None
    movie: Optional[MovieOut] = None

    class Config:
        from_attributes = True

class RecommendationRunOut(BaseModel):
    id: int
    room_id: int
    status: str
    created_at: datetime
    items: List[RunItemOut] = []

    class Config:
        from_attributes = True

class VoteCreate(BaseModel):
    movie_id: int
    vote_value: float # 1 for like, -1 for dislike

class VoteOut(BaseModel):
    id: int
    run_id: int
    user_id: int
    movie_id: int
    vote_value: float
    created_at: datetime
    user: Optional[UserOut] = None

    class Config:
        from_attributes = True

class RunResultOut(BaseModel):
    run_id: int
    winner_movie: Optional[MovieOut] = None
    group_score: Optional[float] = None
    disagreement: Optional[float] = None
    votes: List[VoteOut] = []
    
    class Config:
        from_attributes = True

class GroupRecommendationOut(BaseModel):
    """Schema tạm cho API /recommend/mock"""
    top_movies: List[RecommendedMovie]
    group_score: float
    minimum_score: float
    disagreement: float
    warning: Optional[str] = None
