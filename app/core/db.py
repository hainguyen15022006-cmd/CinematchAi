"""
db.py — Nơi thiết lập kết nối database.
Dùng SQLAlchemy để làm việc với SQLite mà không cần viết SQL thô.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

# check_same_thread=False: cần thiết cho SQLite khi dùng nhiều request đồng thời
engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base là "class gốc" mà mọi model (bảng) sẽ kế thừa
Base = declarative_base()

def get_db():
    """
    Dependency cho FastAPI: mở 1 session DB cho mỗi request,
    và luôn đóng lại sau khi request xong (kể cả khi có lỗi).
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
