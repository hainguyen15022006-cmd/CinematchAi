from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.user import UserOut


class RoomCreate(BaseModel):
    name: Optional[str] = Field(default=None, max_length=120)


class RoomMemberOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    user: Optional[UserOut] = None
    is_ready: bool
    joined_at: datetime

class RoomOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    code: str
    host_id: int
    name: Optional[str] = None
    status: str
    constraints: Optional[str] = None
    created_at: datetime
    members: List[RoomMemberOut] = Field(default_factory=list)


class RoomConstraintsUpdate(BaseModel):
    constraints: str = Field(max_length=4000)  # JSON string for constraints
