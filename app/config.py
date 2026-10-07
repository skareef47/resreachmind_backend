import json
from functools import lru_cache
from typing import Any, List, Optional, Union

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration using Pydantic Settings."""

    APP_NAME: str = "ResearchMind AI"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    CORS_ORIGINS: Union[str, List[str]] = ["http://localhost:3000"]

    # Source provider configurations (Phase 2)
    SEMANTIC_SCHOLAR_BASE_URL: str = "https://api.semanticscholar.org/graph/v1"
    OPENALEX_BASE_URL: str = "https://api.openalex.org"
    OPENALEX_MAILTO: Optional[str] = None
    HTTP_TIMEOUT_SECONDS: float = 10.0
    HTTP_MAX_RETRIES: int = 3
    HTTP_RETRY_BACKOFF_FACTOR: float = 0.5

    SEMANTIC_SCHOLAR_API_KEY: Optional[str] = None
    OPENALEX_API_KEY: Optional[str] = None

    # Hybrid Retrieval configurations (Phase 4)
    RETRIEVAL_CANDIDATE_LIMIT: int = 20
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    SBERT_MODEL_NAME: str = "all-MiniLM-L6-v2"
    RRF_K: int = 60
    RRF_BM25_WEIGHT: float = 1.0
    RRF_SBERT_WEIGHT: float = 1.0

    # Cross-Encoder Reranking configurations (Phase 5)
    CROSS_ENCODER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CROSS_ENCODER_MODEL_VERSION: str = "1.0.0"
    CROSS_ENCODER_BATCH_SIZE: int = 16
    CROSS_ENCODER_DEVICE: str = "cpu"
    CROSS_ENCODER_TOP_K: int = 20
    CROSS_ENCODER_ENABLED: bool = True

    # Database & Persistence configurations (Phase 9)
    DATABASE_URL: str = "sqlite+aiosqlite:///./researchmind.db"
    DATABASE_ECHO: bool = False
    VECTOR_INDEX_DIR: str = "./data/vectors"
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    ENABLE_OBSERVABILITY_METRICS: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("CORS_ORIGINS", mode="after")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse CORS origins from a comma-separated string, JSON string, or list."""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            if v.startswith("[") and v.endswith("]"):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if item]
                except Exception:
                    pass
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        if isinstance(v, (list, tuple, set)):
            return [str(item).strip() for item in v if item]
        return []


@lru_cache()
def get_settings() -> Settings:
    """Retrieve cached application settings instance."""
    return Settings()
