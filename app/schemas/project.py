"""Project schemas."""

from datetime import datetime
import ipaddress
import re
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

from app.schemas.pipeline import PipelineRunStrategy
from app.schemas.author import AuthorCreate
from app.schemas.task import TaskStatusResponse

_DOMAIN_LABEL_RE = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?<!-)$")


def _normalize_and_validate_domain(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise ValueError("Domain is required")

    has_scheme = bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw))
    parsed = urlparse(raw if has_scheme else f"https://{raw}")
    if has_scheme and parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Domain must use http/https if a scheme is provided")

    host = (parsed.hostname or "").strip().rstrip(".")
    if not host:
        raise ValueError("Invalid domain")

    try:
        host_ascii = host.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise ValueError("Invalid domain") from exc

    if len(host_ascii) > 255:
        raise ValueError("Domain is too long")

    try:
        ipaddress.ip_address(host_ascii)
    except ValueError:
        pass
    else:
        raise ValueError("IP addresses are not allowed as project domains")

    labels = host_ascii.split(".")
    if len(labels) < 2:
        raise ValueError("Domain must include a top-level domain (example: example.com)")

    if any(not _DOMAIN_LABEL_RE.match(label) for label in labels):
        raise ValueError("Invalid domain format")

    tld = labels[-1]
    if not re.match(r"^[a-z]{2,63}$", tld):
        raise ValueError("Invalid top-level domain")

    return host_ascii


class ProjectGoals(BaseModel):
    """Business goals for the keyword research project."""

    primary_objective: str = "traffic"
    secondary_goals: list[str] | None = None
    target_monthly_traffic: int | None = None
    target_conversion_rate: float | None = None
    priority_topics: list[str] | None = None
    excluded_topics: list[str] | None = None


class ProjectConstraints(BaseModel):
    """Constraints and limitations."""

    budget_tier: Literal["low", "medium", "high"] = "medium"
    content_team_size: int = 1
    max_keywords_to_target: int | None = None
    max_difficulty_score: float | None = 70
    min_search_volume: int | None = 10
    exclude_branded_keywords: bool = False


class ProjectSettings(BaseModel):
    """Project-specific pipeline settings."""

    skip_steps: list[int] = Field(default_factory=list)
    step_configs: dict[int, dict] | None = None
    notification_webhook: str | None = None
    notification_webhook_secret: str | None = None
    auto_continue_on_error: bool = False


class ProjectCreate(BaseModel):
    """Schema for creating a new project."""

    name: str = Field(min_length=1, max_length=255)
    domain: str = Field(min_length=1, max_length=255)
    description: str | None = None
    primary_language: str = "en"
    primary_locale: str = "en-US"
    secondary_locales: list[str] | None = None
    goals: ProjectGoals | None = None
    constraints: ProjectConstraints | None = None
    settings: ProjectSettings | None = None
    authors: list[AuthorCreate] | None = None

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, value: str) -> str:
        return _normalize_and_validate_domain(value)


class ProjectUpdate(BaseModel):
    """Schema for updating a project."""

    name: str | None = None
    description: str | None = None
    primary_language: str | None = None
    primary_locale: str | None = None
    secondary_locales: list[str] | None = None
    goals: ProjectGoals | None = None
    constraints: ProjectConstraints | None = None
    settings: ProjectSettings | None = None


class ProjectResponse(BaseModel):
    """Schema for project response."""

    id: str
    name: str
    domain: str
    description: str | None
    primary_language: str
    primary_locale: str
    secondary_locales: list[str] | None
    site_maturity: str | None
    primary_goal: str | None
    current_step: int
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ProjectWebhookSecretResponse(BaseModel):
    """Schema for generated project webhook secret."""

    project_id: str
    notification_webhook_secret: str
    updated_at: datetime


class ProjectApiKeyResponse(BaseModel):
    """Schema for generated project integration API key."""

    project_id: str
    api_key: str
    last4: str
    created_at: datetime


class ProjectListResponse(BaseModel):
    """Schema for project list response."""

    items: list[ProjectResponse]
    total: int
    page: int
    page_size: int


class ProjectOnboardingBootstrapRequest(BaseModel):
    """Schema for onboarding bootstrap (create project + start setup pipeline)."""

    name: str = Field(min_length=1, max_length=255)
    domain: str = Field(min_length=1, max_length=255)
    description: str | None = None
    primary_language: str = "en"
    primary_locale: str = "en-US"
    secondary_locales: list[str] | None = None
    goals: ProjectGoals | None = None
    constraints: ProjectConstraints | None = None
    settings: ProjectSettings | None = None
    authors: list[AuthorCreate] | None = Field(
        default=None,
        description="Optional recommended list of author profiles for byline assignment.",
    )
    strategy: PipelineRunStrategy | None = None

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, value: str) -> str:
        return _normalize_and_validate_domain(value)


class ProjectOnboardingBootstrapResponse(BaseModel):
    """Schema returned after onboarding bootstrap kickoff."""

    project: ProjectResponse
    setup_run_id: str
    setup_task: TaskStatusResponse
