from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.db import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.room import Room, RoomMember
from app.schemas.room import RoomCreate, RoomOut, RoomMemberOut, RoomConstraintsUpdate
from app.schemas.run import RecommendationRunOut
from app.services.room_service import create_room, join_room, toggle_ready
from app.services.ai_service import trigger_recommendation

router = APIRouter(prefix="/rooms", tags=["Rooms"])

@router.post("", response_model=RoomOut, status_code=201)
def api_create_room(room_in: RoomCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return create_room(db, host_id=current_user.id, name=room_in.name)

@router.post("/{code}/join", response_model=RoomOut)
def api_join_room(code: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # code is unique room code
    return join_room(db, code=code.upper(), user_id=current_user.id)

@router.post("/{id}/ready", response_model=RoomMemberOut)
def api_toggle_ready(id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return toggle_ready(db, room_id=id, user_id=current_user.id)

@router.put("/{id}/constraints", response_model=RoomOut)
def api_update_constraints(id: int, constraints_in: RoomConstraintsUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    room = db.query(Room).filter(Room.id == id).first()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    if room.host_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only host can update constraints")
    
    room.constraints = constraints_in.constraints
    db.commit()
    db.refresh(room)
    return room

@router.post("/{id}/recommend", response_model=RecommendationRunOut, status_code=201)
def api_trigger_recommend(id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return trigger_recommendation(db, room_id=id, host_id=current_user.id)
