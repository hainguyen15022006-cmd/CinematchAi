from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "CineMatch Backend"
    VERSION: str = "0.1.0"
    
    # Security
    SECRET_KEY: str = "cinematch-demo-secret-change-me"  # In production, change this and read from .env
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day

    # Database
    DATABASE_URL: str = "sqlite:///./cinematch.db"

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:5173",  # React / Vite default
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "*" # Dành cho giai đoạn dev tuần 1 nếu Dương cần test từ thiết bị khác
    ]

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

settings = Settings()
