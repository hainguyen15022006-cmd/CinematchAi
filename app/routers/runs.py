from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.core.db import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.recommendation import RecommendationRun, RunItem, Vote
from app.models.movie import Movie
from app.schemas.run import RunItemOut, VoteCreate, VoteOut, RunResultOut
from app.services.tally_service import tally_votes_and_get_winner

router = APIRouter(prefix="/runs", tags=["Runs & Voting"])

@router.get("/{id}/items", response_model=List[RunItemOut])
def get_run_items(id: int, db: Session = Depends(get_db)):
    items = db.query(RunItem).filter(RunItem.run_id == id).order_by(RunItem.rank).all()
    if not items:
        raise HTTPException(status_code=404, detail="Run items not found")
    return items

@router.post("/{id}/votes", response_model=VoteOut, status_code=201)
def cast_vote(
    id: int, 
    vote_in: VoteCreate, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    run = db.query(RecommendationRun).filter(RecommendationRun.id == id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Optional: check if movie is in the run items
    item = db.query(RunItem).filter(RunItem.run_id == id, RunItem.movie_id == vote_in.movie_id).first()
    if not item:
        raise HTTPException(status_code=400, detail="Movie is not in this recommendation run")

    existing_vote = db.query(Vote).filter(Vote.run_id == id, Vote.user_id == current_user.id, Vote.movie_id == vote_in.movie_id).first()
    if existing_vote:
        existing_vote.vote_value = vote_in.vote_value
        db.commit()
        db.refresh(existing_vote)
        return existing_vote
    else:
        new_vote = Vote(
            run_id=id,
            user_id=current_user.id,
            movie_id=vote_in.movie_id,
            vote_value=vote_in.vote_value
        )
        db.add(new_vote)
        db.commit()
        db.refresh(new_vote)
        return new_vote

@router.get("/{id}/result", response_model=RunResultOut)
def get_run_result(id: int, db: Session = Depends(get_db)):
    # This recalculates or just fetches the result.
    run = tally_votes_and_get_winner(db, run_id=id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    return run
