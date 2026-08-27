from pydantic import BaseModel, EmailStr
from typing import Optional

class UserOut(BaseModel):
    id: int
    email: EmailStr

    class Config:
        from_attributes = True

class UserProfileOut(UserOut):
    preferences_text: Optional[str] = None

class UserPreferenceUpdate(BaseModel):
    preferences_text: str
