from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.db import Base

class Movie(Base):
    __tablename__ = "movies"

    id = Column(Integer, primary_key=True, index=True)
    movielens_id = Column(Integer, unique=True, index=True, nullable=False)
    title = Column(String, nullable=False)
    genres = Column(String, nullable=True)  # "Action|Comedy"
    release_year = Column(Integer, nullable=True)
    imdb_url = Column(String, nullable=True)

    ratings = relationship("Rating", back_populates="movie")
    run_items = relationship("RunItem", back_populates="movie")


class Rating(Base):
    __tablename__ = "ratings"
    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="uq_rating_user_movie"),
        CheckConstraint("rating >= 1 AND rating <= 5", name="ck_rating_range"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False)
    rating = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="ratings")
    movie = relationship("Movie", back_populates="ratings")
