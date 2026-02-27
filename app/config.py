"""Application configuration using pydantic-settings."""

import base64
import hashlib
import json
from ast import literal_eval
from functools import lru_cache
from typing import Annotated, ClassVar, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


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
    internal_api_prefix: str = ""
    integration_api_prefix: str = "/integration"
    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
    ]

    # External integration API key auth
    integration_api_keys: str = ""
    integration_api_key_pepper: str | None = None

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
    public_api_base_url: str | None = None

    # Transactional email (Resend)
    resend_api_key: str | None = None
    resend_from_email: str = "DonkeySEO <onboarding@donkey.support>" # Change this when going for the resend PRO subscription
    email_verification_callback_url: str | None = None
    email_verification_token_expire_hours: int = 48

    # Stripe billing
    stripe_secret_key: str | None = None
    stripe_publishable_key: str | None = None
    stripe_webhook_secret: str | None = None
    stripe_product_donkeyseo: str | None = None
    stripe_price_starter_monthly: str | None = None
    stripe_price_starter_yearly: str | None = None
    stripe_price_growth_monthly: str | None = None
    stripe_price_growth_yearly: str | None = None
    stripe_price_agency_monthly: str | None = None
    stripe_price_agency_yearly: str | None = None
    stripe_price_article_addon: str | None = None
    stripe_free_trial_days: int = 0
    webhook_secret_encryption_key: str | None = None

    # LLM Configuration
    default_llm_model: str = "openai:gpt-4-turbo"
    openrouter_api_key: str | None = None
    embeddings_model: str = "qwen/qwen3-embedding-8b"
    embeddings_provider: str | None = "nebius"
    embeddings_allow_fallbacks: bool = False
    llm_max_retries: int = 3
    llm_timeout_fast: int = 120
    llm_timeout_standard: int = 300
    llm_timeout_reasoning: int = 600

    def get_llm_timeout(self, tier: str = "standard") -> int:
        """Return the LLM timeout in seconds for a given model tier."""
        return getattr(self, f"llm_timeout_{tier}", self.llm_timeout_standard)

    # Dynamic model selector (OpenRouter/Arena-driven)
    model_selector_enabled: bool = False
    model_selector_snapshot_path: str = "app/agents/model_selection_snapshot.json"
    model_selector_max_price_dev: float = 0.0
    model_selector_max_price_staging: float = 0.0
    model_selector_max_price_prod: float = 0.0
    model_selector_arena_weight: float = 0.25
    model_selector_openrouter_weight: float = 0.75
    model_selector_fallback_model: str = "openrouter:google/gemma-3-27b-it:free"

    # Per-tier model overrides (optional — override the built-in defaults below)
    dev_model_reasoning: str | None = None
    dev_model_standard: str | None = None
    dev_model_fast: str | None = None
    prod_model_reasoning: str | None = None
    prod_model_standard: str | None = None
    prod_model_fast: str | None = None

    _MODEL_DEFAULTS: ClassVar[dict[str, dict[str, str]]] = {
        "development": {
            "reasoning": "openrouter:minimax/minimax-m2.5",
            "standard": "openrouter:minimax/minimax-m2.5",
            "fast": "openrouter:minimax/minimax-m2.5",
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

    def get_model_selector_max_price(self, environment: str | None = None) -> float:
        """Resolve model-selector max price for a target environment."""
        target_environment = environment or self.environment
        if target_environment == "development":
            return self.model_selector_max_price_dev
        if target_environment == "staging":
            return self.model_selector_max_price_staging
        return self.model_selector_max_price_prod

    # DataForSEO
    dataforseo_login: str | None = None
    dataforseo_password: str | None = None

    # Cloudflare R2 (private brand assets)
    cloudflare_r2_account_id: str | None = None
    cloudflare_r2_access_key_id: str | None = None
    cloudflare_r2_secret_access_key: str | None = None
    cloudflare_r2_bucket: str | None = None
    cloudflare_r2_region: str = "auto"
    brand_asset_max_bytes: int = 5_000_000
    brand_assets_max_count: int = 8
    signed_url_ttl_seconds: int = 90
    content_image_width: int = 1200
    content_image_height: int = 630
    content_image_render_timeout_ms: int = 30_000
    content_image_retry_attempts: int = 3
    content_image_retry_backoff_ms: int = 750

    setup_pipeline_task_workers: int = 2
    setup_pipeline_task_queue_size: int = 100
    discovery_pipeline_task_workers: int = 2
    discovery_pipeline_task_queue_size: int = 200
    content_pipeline_task_workers: int = 3
    content_pipeline_task_queue_size: int = 300

    @property
    def stripe_enabled(self) -> bool:
        """Whether Stripe billing is configured."""
        return bool(self.stripe_secret_key)

    def get_integration_api_keys(self) -> set[str]:
        """Return configured integration API keys from comma-separated env value."""
        return {
            value.strip()
            for value in self.integration_api_keys.split(",")
            if value and value.strip()
        }

    @field_validator("database_url", mode="before")
    @classmethod
    def _normalize_database_url(cls, value: object) -> object:
        """Normalize sync Postgres URLs to asyncpg for async SQLAlchemy usage."""
        if not isinstance(value, str):
            return value

        raw = value.strip()
        if raw.startswith("postgresql+asyncpg://"):
            return raw
        if raw.startswith("postgresql+psycopg2://"):
            return raw.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
        if raw.startswith("postgresql+psycopg://"):
            return raw.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
        if raw.startswith("postgresql://"):
            return raw.replace("postgresql://", "postgresql+asyncpg://", 1)
        if raw.startswith("postgres://"):
            return raw.replace("postgres://", "postgresql+asyncpg://", 1)
        return raw

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: object) -> object:
        """Accept JSON list/string or comma-separated values for CORS_ORIGINS."""
        def normalize(origin: object) -> str:
            cleaned = str(origin).strip().strip("'\"")
            if cleaned.startswith("[") and cleaned.endswith("]"):
                cleaned = cleaned[1:-1].strip().strip("'\"")
            return cleaned

        if isinstance(value, list):
            return [normalize(origin) for origin in value if normalize(origin)]
        if not isinstance(value, str):
            return value

        raw = value.strip()
        if not raw:
            return []

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed = literal_eval(raw)
            except (ValueError, SyntaxError):
                parsed = [normalize(origin) for origin in raw.split(",")]
                return [origin for origin in parsed if origin]

        if isinstance(parsed, str):
            parsed = [parsed]
        if isinstance(parsed, tuple | set):
            parsed = list(parsed)
        if not isinstance(parsed, list):
            raise ValueError(
                "CORS_ORIGINS must be a JSON array, JSON string, or comma-separated string.",
            )
        return [normalize(origin) for origin in parsed if normalize(origin)]

    @staticmethod
    def _normalize_route_prefix(prefix: str) -> str:
        """Normalize route prefix values to `/segment` form."""
        normalized = prefix.strip()
        if not normalized or normalized == "/":
            return ""
        return f"/{normalized.strip('/')}"

    @staticmethod
    def _join_route_prefixes(*prefixes: str) -> str:
        """Join route prefixes while preventing duplicate slashes."""
        segments: list[str] = []
        for prefix in prefixes:
            normalized = Settings._normalize_route_prefix(prefix)
            if not normalized:
                continue
            segments.extend(segment for segment in normalized.split("/") if segment)
        return f"/{'/'.join(segments)}" if segments else ""

    @property
    def resolved_internal_api_prefix(self) -> str:
        """Resolve internal API router prefix."""
        return  self._join_route_prefixes(self.api_v1_prefix, self.internal_api_prefix)

    @property
    def versioned_integration_api_prefix(self) -> str:
        """Resolve integration mount path under the API v1 namespace."""
        return self._join_route_prefixes(self.api_v1_prefix, self.integration_api_prefix)

    def get_integration_api_key_pepper(self) -> bytes:
        """Return server-side pepper bytes used for API key HMAC hashing."""
        pepper_material = self.integration_api_key_pepper or self.jwt_secret_key
        scoped = f"integration-api-key:{pepper_material}"
        return scoped.encode("utf-8")

    def get_webhook_secret_encryption_key(self) -> bytes:
        """Return a Fernet-compatible key for webhook-secret field encryption."""
        key_material = self.webhook_secret_encryption_key or self.jwt_secret_key
        scoped = f"webhook-secret:{key_material}"
        digest = hashlib.sha256(scoped.encode("utf-8")).digest()
        return base64.urlsafe_b64encode(digest)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()

    return settings


settings = get_settings()
