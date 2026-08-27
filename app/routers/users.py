from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.db import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.schemas.user import UserProfileOut, UserPreferenceUpdate

router = APIRouter(prefix="/users", tags=["Users"])

@router.get("/me", response_model=UserProfileOut)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@router.put("/me/preferences", response_model=UserProfileOut)
def update_preferences(
    prefs_in: UserPreferenceUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    current_user.preferences_text = prefs_in.preferences_text
    db.commit()
    db.refresh(current_user)
    return current_user
