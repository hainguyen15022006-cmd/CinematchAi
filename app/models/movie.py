from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
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

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False)
    rating = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="ratings")
    movie = relationship("Movie", back_populates="ratings")
