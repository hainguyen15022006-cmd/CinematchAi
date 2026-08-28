from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.movie import Movie
from app.models.recommendation import RecommendationRun, RunItem
from app.models.room import Room, RoomMember


def _mock_ai_score(movielens_id: int) -> float:
    """Return a reproducible placeholder score until the AI adapter is wired."""
    return round(3.0 + (movielens_id % 20) / 10, 2)


def trigger_recommendation(db: Session, room_id: int, host_id: int) -> RecommendationRun:
    room = db.query(Room).filter(Room.id == room_id).first()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    if room.host_id != host_id:
        raise HTTPException(
            status_code=403,
            detail="Only host can trigger recommendation",
        )
    if room.status != "OPEN":
        raise HTTPException(status_code=409, detail="Room is not open")

    # Check if all members are ready
    members = db.query(RoomMember).filter(RoomMember.room_id == room_id).all()
    if not members or not all(member.is_ready for member in members):
        raise HTTPException(status_code=400, detail="Not all members are ready")

    movies = db.query(Movie).order_by(Movie.movielens_id).limit(10).all()
    if not movies:
        raise HTTPException(
            status_code=409,
            detail="Movie catalog is empty; run scripts/seed_movies.py first",
        )

    # Create run
    run = RecommendationRun(room_id=room_id, status="VOTING")
    db.add(run)
    db.flush()

    room.status = "RUNNING"

    # Temporary deterministic adapter. It will be replaced by the AI service.
    for rank, movie in enumerate(movies, start=1):
        item = RunItem(
            run_id=run.id,
            movie_id=movie.id,
            rank=rank,
            ai_score=_mock_ai_score(movie.movielens_id),
        )
        db.add(item)

    db.commit()
    db.refresh(run)
    return run
