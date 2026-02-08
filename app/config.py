"""Application configuration using pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "DonkeySEO"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: Literal["development", "staging", "production"] = "development"

    # API
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/donkeyseo"
    database_pool_size: int = 5
    database_max_overflow: int = 10
    database_echo: bool = False

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 86400  # 24 hours

    # JWT Authentication
    jwt_secret_key: str = "change-me-in-production-use-a-real-secret-key"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # LLM Configuration
    default_llm_model: str = "openai:gpt-4-turbo"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    llm_max_retries: int = 3
    llm_timeout_seconds: int = 60

    # DataForSEO
    dataforseo_login: str | None = None
    dataforseo_password: str | None = None

    # Rate Limiting
    keyword_api_calls_per_minute: int = 100
    serp_api_calls_per_minute: int = 60
    llm_calls_per_minute: int = 50

    # Pipeline Defaults
    default_skip_steps: list[int] = [8, 9, 10, 11, 14]
    max_keywords_per_project: int = 50000
    batch_size_keyword_enrichment: int = 100


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
