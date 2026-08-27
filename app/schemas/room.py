from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from app.schemas.user import UserOut

class RoomCreate(BaseModel):
    name: Optional[str] = None

class RoomMemberOut(BaseModel):
    id: int
    user_id: int
    user: Optional[UserOut] = None
    is_ready: bool
    joined_at: datetime

    class Config:
        from_attributes = True

class RoomOut(BaseModel):
    id: int
    code: str
    host_id: int
    name: Optional[str] = None
    status: str
    constraints: Optional[str] = None
    created_at: datetime
    members: List[RoomMemberOut] = []

    class Config:
        from_attributes = True

class RoomConstraintsUpdate(BaseModel):
    constraints: str # JSON string for constraints
