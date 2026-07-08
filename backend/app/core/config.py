from functools import lru_cache
from pydantic import Field
from pydantic import BaseSettings
from pydantic import SettingsConfigDict
class Setting(BaseSettings):
    # Application configuration loaded from environment variable
    APP_NAME: str = "CA Mentor AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    Host: str = "0.0.0.0"
    PORT: int = 8000

    # API
    API_V1_PREFIX: str = "/api/v1"

    # Database
    DATABASE_URL: str = ""

    # Qdrant
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""

    # Google Gemini
    GOOGLE_API_KEY: str = ""

    # AI MODELS
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

@lru_cache()
def get_settings() -> Setting:
    # return the cached settings instance
    return Setting()

settings = get_settings()