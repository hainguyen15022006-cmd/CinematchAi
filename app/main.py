"""
main.py — Entry point of the whole Backend.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.db import Base, SessionLocal, engine, ensure_runtime_schema
from app.routers import (
    auth_router,
    movies_router,
    recommendations_router,
    rooms_router,
    runs_router,
    users_router,
)
from app.services.catalog_service import seed_movies_if_empty

LOGGER = logging.getLogger(__name__)

# Create tables if not exists, then add columns introduced later.
Base.metadata.create_all(bind=engine)
ensure_runtime_schema()

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

# CORS middleware. BACKEND_CORS_ORIGIN_REGEX additionally allows the
# generated preview hostnames of one deployed Frontend project.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_origin_regex=settings.BACKEND_CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def seed_catalog_on_startup() -> None:
    """Restore the movie catalog after a deployment with an empty disk."""
    if not settings.AUTO_SEED_MOVIES:
        return
    with SessionLocal() as db:
        inserted = seed_movies_if_empty(db)
    if inserted:
        LOGGER.info("Seeded %d movies into an empty catalog", inserted)


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
