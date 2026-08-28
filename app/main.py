"""
main.py — Điểm khởi động của toàn bộ Backend.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.db import Base, engine
from app.routers import (
    auth_router,
    movies_router,
    recommendations_router,
    rooms_router,
    runs_router,
    users_router,
)

# Create tables if not exists
Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}

# Include routers
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(movies_router)
app.include_router(recommendations_router)
app.include_router(rooms_router)
app.include_router(runs_router)
