import random
import string

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.room import Room, RoomMember
from cinematch.recommendation.group import MAX_GROUP_SIZE


def generate_unique_room_code(db: Session) -> str:
    while True:
        code = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
        if not db.query(Room).filter(Room.code == code).first():
            return code


def create_room(db: Session, host_id: int, name: str | None = None) -> Room:
    code = generate_unique_room_code(db)
    room = Room(code=code, host_id=host_id, name=name)
    db.add(room)
    db.flush()

    member = RoomMember(room_id=room.id, user_id=host_id, is_ready=False)
    db.add(member)
    db.commit()
    db.refresh(room)
    return room


def join_room(db: Session, code: str, user_id: int) -> Room:
    room = db.query(Room).filter(Room.code == code).first()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    if room.status != "OPEN":
        raise HTTPException(status_code=409, detail="Room is not open")

    existing_member = db.query(RoomMember).filter(
        RoomMember.room_id == room.id,
        RoomMember.user_id == user_id,
    ).first()
    if existing_member:
        return room

    member_count = db.query(RoomMember).filter(
        RoomMember.room_id == room.id
    ).count()
    if member_count >= MAX_GROUP_SIZE:
        raise HTTPException(
            status_code=409,
            detail=f"Room already has maximum {MAX_GROUP_SIZE} members",
        )

    member = RoomMember(room_id=room.id, user_id=user_id, is_ready=False)
    db.add(member)
    db.commit()

    return room


def toggle_ready(db: Session, room_id: int, user_id: int) -> RoomMember:
    member = db.query(RoomMember).filter(
        RoomMember.room_id == room_id,
        RoomMember.user_id == user_id,
    ).first()
    if not member:
        raise HTTPException(status_code=404, detail="Not a member of this room")

    member.is_ready = not member.is_ready
    db.commit()
    db.refresh(member)
    return member
