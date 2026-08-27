from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.db import Base

class Room(Base):
    __tablename__ = "rooms"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(6), unique=True, index=True, nullable=False)
    host_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=True)
    status = Column(String, default="OPEN") # OPEN, READY, RUNNING, FINISHED
    constraints = Column(Text, nullable=True) # JSON string
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    host = relationship("User")
    members = relationship("RoomMember", back_populates="room")
    runs = relationship("RecommendationRun", back_populates="room")


class RoomMember(Base):
    __tablename__ = "room_members"
    __table_args__ = (
        UniqueConstraint("room_id", "user_id", name="uq_room_member"),
    )

    id = Column(Integer, primary_key=True, index=True)
    room_id = Column(Integer, ForeignKey("rooms.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    is_ready = Column(Boolean, default=False)
    joined_at = Column(DateTime(timezone=True), server_default=func.now())

    room = relationship("Room", back_populates="members")
    user = relationship("User", back_populates="room_memberships")
