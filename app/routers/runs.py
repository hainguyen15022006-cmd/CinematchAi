from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.security import get_current_user
from app.models.movie import Movie
from app.models.recommendation import RecommendationRun, RunItem, Vote
from app.models.room import RoomMember
from app.models.user import User
from app.schemas.run import RunItemOut, RunResultOut, VoteCreate, VoteOut
from app.services.tally_service import tally_votes_and_get_winner

router = APIRouter(prefix="/runs", tags=["Runs & Voting"])


def _get_run_for_member(
    run_id: int,
    user_id: int,
    db: Session,
) -> RecommendationRun:
    run = (
        db.query(RecommendationRun)
        .filter(RecommendationRun.id == run_id)
        .first()
    )
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    membership = db.query(RoomMember).filter(
        RoomMember.room_id == run.room_id,
        RoomMember.user_id == user_id,
    ).first()
    if membership is None:
        raise HTTPException(status_code=403, detail="Not a member of this room")
    return run


@router.get("/{id}/items", response_model=List[RunItemOut])
def get_run_items(
    id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _get_run_for_member(id, current_user.id, db)
    items = (
        db.query(RunItem)
        .filter(RunItem.run_id == id)
        .order_by(RunItem.rank)
        .all()
    )
    if not items:
        raise HTTPException(status_code=404, detail="Run items not found")
    return items


@router.post("/{id}/votes", response_model=VoteOut, status_code=201)
def cast_vote(
    id: int,
    vote_in: VoteCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    run = _get_run_for_member(id, current_user.id, db)
    if run.status != "VOTING":
        raise HTTPException(status_code=409, detail="Voting is closed")

    movie = db.query(Movie).filter(
        Movie.movielens_id == vote_in.movie_id
    ).first()
    if movie is None:
        raise HTTPException(status_code=404, detail="Movie not found")

    item = db.query(RunItem).filter(
        RunItem.run_id == id,
        RunItem.movie_id == movie.id,
    ).first()
    if not item:
        raise HTTPException(
            status_code=400,
            detail="Movie is not in this recommendation run",
        )

    existing_vote = db.query(Vote).filter(
        Vote.run_id == id,
        Vote.user_id == current_user.id,
    ).first()
    if existing_vote:
        existing_vote.movie = movie
        existing_vote.vote_value = vote_in.vote_value
        db.commit()
        db.refresh(existing_vote)
        return existing_vote
    new_vote = Vote(
        run_id=id,
        user_id=current_user.id,
        movie=movie,
        vote_value=vote_in.vote_value,
    )
    db.add(new_vote)
    db.commit()
    db.refresh(new_vote)
    return new_vote


@router.post("/{id}/finalize", response_model=RunResultOut)
def finalize_run(
    id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    run = _get_run_for_member(id, current_user.id, db)
    if run.room.host_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only host can finalize voting")
    if run.status == "FINISHED":
        return run

    run = tally_votes_and_get_winner(db, run_id=id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/{id}/result", response_model=RunResultOut)
def get_run_result(
    id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    run = _get_run_for_member(id, current_user.id, db)
    if run.status != "FINISHED":
        raise HTTPException(status_code=409, detail="Voting has not been finalized")
    return run
