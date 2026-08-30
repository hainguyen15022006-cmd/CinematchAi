"""
db.py — Database connection setup.
Uses SQLAlchemy to work with SQLite without writing raw SQL.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

# check_same_thread=False: required for SQLite when handling concurrent requests
connect_args = (
    {"check_same_thread": False}
    if settings.DATABASE_URL.startswith("sqlite")
    else {}
)
engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base is the declarative base class that every model (table) inherits from
Base = declarative_base()

def get_db():
    """
    FastAPI dependency: opens one DB session per request and always
    closes it once the request finishes (even when an error occurs).
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
