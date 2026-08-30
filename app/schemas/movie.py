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
        description="Original MovieLens ID, not the internal database key",
    )
    rating: float = Field(
        ge=1,
        le=5,
        description="Rating value, must be within the 1-5 range",
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
