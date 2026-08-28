from sqlalchemy import (
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

class RecommendationRun(Base):
    __tablename__ = "recommendation_runs"

    id = Column(Integer, primary_key=True, index=True)
    room_id = Column(Integer, ForeignKey("rooms.id"), nullable=False)
    status = Column(String, default="PENDING") # PENDING, VOTING, FINISHED
    winner_movie_id = Column(Integer, ForeignKey("movies.id"), nullable=True)
    group_score = Column(Float, nullable=True)
    disagreement = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    room = relationship("Room", back_populates="runs")
    items = relationship("RunItem", back_populates="run")
    votes = relationship("Vote", back_populates="run")
    winner_movie = relationship("Movie")


class RunItem(Base):
    __tablename__ = "run_items"
    __table_args__ = (
        UniqueConstraint("run_id", "movie_id", name="uq_run_item_movie"),
        UniqueConstraint("run_id", "rank", name="uq_run_item_rank"),
    )

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("recommendation_runs.id"), nullable=False)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False)
    rank = Column(Integer, nullable=False)
    ai_score = Column(Float, nullable=True)

    run = relationship("RecommendationRun", back_populates="items")
    movie = relationship("Movie", back_populates="run_items")


class Vote(Base):
    __tablename__ = "votes"
    __table_args__ = (
        UniqueConstraint("run_id", "user_id", name="uq_vote_run_user"),
    )

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("recommendation_runs.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False)
    vote_value = Column(Float, nullable=False) # e.g. 1 (Like), -1 (Dislike)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    run = relationship("RecommendationRun", back_populates="votes")
    user = relationship("User", back_populates="votes")
    movie = relationship("Movie")
