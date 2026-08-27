from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.db import get_db
from app.core.security import hash_password, verify_password, create_access_token
from app.models.user import User
from app.schemas.auth import UserCreate, Token
from app.schemas.user import UserOut

router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/register", response_model=UserOut, status_code=201)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == user_in.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = User(
        email=user_in.email,
        password_hash=hash_password(user_in.password),
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@router.post("/login", response_model=Token)
def login(user_in: UserCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == user_in.email).first()
    if not user or not verify_password(user_in.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"user_id": user.id})
    return {"access_token": token, "token_type": "bearer"}
