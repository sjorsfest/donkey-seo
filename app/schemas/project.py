"""Project schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


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


class ProjectListResponse(BaseModel):
    """Schema for project list response."""

    items: list[ProjectResponse]
    total: int
    page: int
    page_size: int
