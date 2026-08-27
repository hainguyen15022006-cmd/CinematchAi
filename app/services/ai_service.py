import random
from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.models.room import Room, RoomMember
from app.models.movie import Movie
from app.models.recommendation import RecommendationRun, RunItem

def trigger_recommendation(db: Session, room_id: int, host_id: int) -> RecommendationRun:
    room = db.query(Room).filter(Room.id == room_id).first()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    
    if room.host_id != host_id:
        raise HTTPException(status_code=403, detail="Only host can trigger recommendation")
    
    # Check if all members are ready
    members = db.query(RoomMember).filter(RoomMember.room_id == room_id).all()
    if not all(m.is_ready for m in members):
        raise HTTPException(status_code=400, detail="Not all members are ready")
    
    # Create run
    run = RecommendationRun(room_id=room_id, status="VOTING")
    db.add(run)
    db.commit()
    db.refresh(run)
    
    # Update room status
    room.status = "RUNNING"
    db.commit()

    # Generate top 10 mock items
    movies = db.query(Movie).limit(10).all() # Just getting first 10 for mock
    for i, m in enumerate(movies):
        score = round(random.uniform(3.0, 5.0), 2)
        item = RunItem(run_id=run.id, movie_id=m.id, rank=i+1, ai_score=score)
        db.add(item)
    
    db.commit()
    db.refresh(run)
    return run
