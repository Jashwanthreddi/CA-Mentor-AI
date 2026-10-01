from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration.

    Values are loaded from environment variables and .env.
    Secrets must never be hard-coded in source code.
    """

    # ============================================================
    # APPLICATION
    # ============================================================

    APP_NAME: str = "CA Mentor AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ============================================================
    # API
    # ============================================================

    API_V1_PREFIX: str = "/api/v1"

    # ============================================================
    # DATABASE
    # ============================================================

    DATABASE_URL: str = ""

    # ============================================================
    # QDRANT
    # ============================================================

    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "ca_mentor_chunks"

    # ============================================================
    # GOOGLE GEMINI
    # ============================================================

    GOOGLE_API_KEY: str = ""

    # ============================================================
    # AI MODELS
    # ============================================================

    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"

    # ============================================================
    # PYDANTIC SETTINGS
    # ============================================================

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Return the cached application settings.
    """
    return Settings()


settings = get_settings()