from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr

class UserProfileOut(UserOut):
    preferences_text: Optional[str] = None


class UserPreferenceUpdate(BaseModel):
    preferences_text: str = Field(max_length=1000)
