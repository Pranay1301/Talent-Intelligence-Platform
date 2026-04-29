"""
Application configuration using pydantic-settings.
Loads environment variables from .env file.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = "sqlite:///./resume_screening.db"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # Models
    embedding_model: str = "all-MiniLM-L6-v2"
    spacy_model: str = "en_core_web_sm"

    # FAISS
    faiss_index_path: str = "./faiss_index/index.faiss"
    faiss_metadata_path: str = "./faiss_index/metadata.json"

    # JWT
    secret_key: str = "dev-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Streamlit
    streamlit_server_port: int = 8501
    api_base_url: str = "http://localhost:8000"

    # Upload
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 10
    allowed_extensions: list[str] = [".pdf", ".docx"]

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Create upload directory if it doesn't exist
settings = get_settings()
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(os.path.dirname(settings.faiss_index_path), exist_ok=True)
