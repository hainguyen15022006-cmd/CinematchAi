from datetime import datetime
from typing import Optional

from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field


class MovieOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    movielens_id: int
    title: str
    genres: Optional[str] = None
    release_year: Optional[int] = None
    imdb_url: Optional[str] = None

class RatingCreate(BaseModel):
    movie_id: int = Field(
        gt=0,
        description="ID gốc MovieLens, không phải khóa nội bộ database",
    )
    rating: float = Field(
        ge=1,
        le=5,
        description="Điểm rating, bắt buộc trong khoảng 1-5",
    )

class RatingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    movie_id: int = Field(
        validation_alias=AliasChoices(
            AliasPath("movie", "movielens_id"),
            "movie_id",
        )
    )
    rating: float
    created_at: datetime
