from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class MovieOut(BaseModel):
    id: int
    movielens_id: int
    title: str
    genres: Optional[str] = None
    release_year: Optional[int] = None
    imdb_url: Optional[str] = None

    class Config:
        from_attributes = True

class RatingCreate(BaseModel):
    movie_id: int
    rating: float = Field(ge=1, le=5, description="Điểm rating, bắt buộc trong khoảng 1-5")

class RatingOut(BaseModel):
    id: int
    movie_id: int
    rating: float
    created_at: datetime

    class Config:
        from_attributes = True
