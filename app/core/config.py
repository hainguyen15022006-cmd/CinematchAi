from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# app/core/config.py -> app/core -> app -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    PROJECT_NAME: str = "CineMatch Backend"
    VERSION: str = "0.1.0"

    # Security
    SECRET_KEY: str = "cinematch-demo-secret-change-me"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day

    # Database
    DATABASE_URL: str = "sqlite:///./cinematch.db"

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:5173",  # React / Vite default
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]
    # Lets preview deployments of one frontend project through without
    # listing every generated hostname, for example:
    # BACKEND_CORS_ORIGIN_REGEX=https://.*\.vercel\.app
    BACKEND_CORS_ORIGIN_REGEX: Optional[str] = None

    # Model serving. Paths are relative to the repository root unless
    # they are absolute. The bundle is created by
    # scripts/build_serving_bundle.py.
    SERVING_ARTIFACTS_DIR: str = "artifacts/serving"
    SERVING_CHECKPOINT_FILE: str = "mf_checkpoint.pt"
    SERVING_TRAIN_RATINGS_FILE: str = "train_ratings.csv.gz"
    SERVING_ID_MAPPINGS_FILE: str = "id_mappings.json"
    SERVING_MOVIES_FILE: str = "movies.csv.gz"

    # Seed the movie catalog on startup when the database is still empty.
    # A free hosting plan usually resets its disk on every deploy, so the
    # catalog has to be restored without a manual step.
    AUTO_SEED_MOVIES: bool = True

    # Group recommendation defaults used when a room sets no constraints.
    RECOMMENDER_NEIGHBOR_COUNT: int = 20
    RECOMMENDER_TOP_K: int = 10
    RECOMMENDER_STRATEGY: str = "average"
    RECOMMENDER_MISERY_THRESHOLD: float = 2.0

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT


settings = Settings()
