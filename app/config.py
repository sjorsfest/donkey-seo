"""Application configuration using pydantic-settings."""

from functools import lru_cache
from typing import ClassVar, Literal

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

    # OAuth (Google + Twitter)
    google_client_id: str | None = None
    google_client_secret: str | None = None
    google_callback_url: str | None = None
    twitter_client_id: str | None = None
    twitter_client_secret: str | None = None
    twitter_callback_url: str | None = None
    oauth_state_secret: str = "change-me-oauth-state-secret"

    # LLM Configuration
    default_llm_model: str = "openai:gpt-4-turbo"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    openrouter_api_key: str | None = None
    llm_max_retries: int = 3
    llm_timeout_seconds: int = 60

    # Per-tier model overrides (optional — override the built-in defaults below)
    dev_model_reasoning: str | None = None
    dev_model_standard: str | None = None
    dev_model_fast: str | None = None
    prod_model_reasoning: str | None = None
    prod_model_standard: str | None = None
    prod_model_fast: str | None = None

    _MODEL_DEFAULTS: ClassVar[dict[str, dict[str, str]]] = {
        "development": {
            "reasoning": "openrouter:google/gemma-3-27b-it:free",
            "standard": "openrouter:google/gemma-3-27b-it:free",
            "fast": "openrouter:google/gemma-3-27b-it:free",
        },
        "staging": {
            "reasoning": "openrouter:google/gemma-3-27b-it:free",
            "standard": "openrouter:google/gemma-3-27b-it:free",
            "fast": "openrouter:google/gemma-3-27b-it:free",
        },
        "production": {
            "reasoning": "anthropic:claude-sonnet-4-5",
            "standard": "anthropic:claude-sonnet-4-5",
            "fast": "anthropic:claude-sonnet-4-5",
        },
    }

    def get_model(self, tier: str = "standard") -> str:
        """Resolve the model string for a given tier based on environment.

        Priority: env var override > built-in defaults > default_llm_model fallback.
        """
        env_prefix = "dev" if self.environment in ("development", "staging") else "prod"
        override = getattr(self, f"{env_prefix}_model_{tier}", None)
        if isinstance(override, str) and override:
            return override

        env_defaults = self._MODEL_DEFAULTS.get(self.environment, {})
        resolved = env_defaults.get(tier, self.default_llm_model)
        if isinstance(resolved, str):
            return resolved
        return self.default_llm_model

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
    settings = Settings()

    return settings


settings = get_settings()
