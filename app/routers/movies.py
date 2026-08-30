from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.core.db import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.movie import Movie, Rating
from app.schemas.movie import MovieOut, RatingCreate, RatingOut

router = APIRouter(tags=["Movies & Ratings"])


@router.get("/movies", response_model=List[MovieOut])
def list_movies(limit: int = 50, skip: int = 0, db: Session = Depends(get_db)):
    return db.query(Movie).offset(skip).limit(limit).all()


@router.get("/ratings", response_model=List[RatingOut])
def list_current_user_ratings(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return saved ratings so onboarding can restore state after reload."""
    return (
        db.query(Rating)
        .filter(Rating.user_id == current_user.id)
        .order_by(Rating.movie_id)
        .all()
    )


@router.post("/ratings", response_model=RatingOut, status_code=201)
def create_rating(
    rating_in: RatingCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    movie = db.query(Movie).filter(
        Movie.movielens_id == rating_in.movie_id
    ).first()
    if movie is None:
        raise HTTPException(status_code=404, detail="Movie not found")

    saved_rating = db.query(Rating).filter(
        Rating.user_id == current_user.id,
        Rating.movie_id == movie.id,
    ).first()
    if saved_rating is None:
        saved_rating = Rating(
            user_id=current_user.id,
            movie=movie,
            rating=rating_in.rating,
        )
        db.add(saved_rating)
    else:
        saved_rating.rating = rating_in.rating

    db.commit()
    db.refresh(saved_rating)
    return saved_rating
