"""Pydantic schemas for external integration API responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator


class IntegrationIndexResponse(BaseModel):
    """Top-level integration API index."""

    service: str
    docs_path: str
    openapi_path: str
    guide_path: str
    guide_markdown_path: str
    article_list_path_template: str
    pillar_list_path_template: str
    article_latest_path_template: str
    article_version_path_template: str
    article_publication_patch_path_template: str
    auth_header: str


class IntegrationGuideResponse(BaseModel):
    """Detailed implementation guide payload."""

    title: str
    schema_version: str
    markdown: str
    modular_document_contract: dict[str, Any]
    modular_document_field_reference: dict[str, Any]
    block_type_reference: dict[str, Any]
    webhook_contract: dict[str, Any]
    client_env_vars: dict[str, str]
    client_env_template: str


class IntegrationArticleVersionResponse(BaseModel):
    """External API response for article version retrieval."""

    id: str
    article_id: str
    project_id: str
    version_number: int
    title: str
    slug: str
    primary_keyword: str
    modular_document: dict[str, Any]
    rendered_html: str
    qa_report: dict[str, Any] | None
    status: str
    change_reason: str | None
    generation_model: str | None
    generation_temperature: float | None
    created_by_regeneration: bool
    created_at: datetime
    updated_at: datetime


class IntegrationPillarReference(BaseModel):
    """Pillar reference payload for integration clients."""

    id: str
    name: str
    slug: str


class IntegrationPillarResponse(BaseModel):
    """Pillar list payload for integration clients."""

    id: str
    project_id: str
    name: str
    slug: str
    description: str | None
    status: str
    source: str
    locked: bool
    primary_brief_count: int
    secondary_brief_count: int
    total_brief_count: int
    primary_article_count: int
    published_primary_article_count: int
    created_at: datetime
    updated_at: datetime


class IntegrationArticleSummaryResponse(BaseModel):
    """Lightweight article summary for listing integrations."""

    id: str
    project_id: str
    brief_id: str
    title: str
    slug: str
    primary_keyword: str
    current_version: int
    status: str
    publish_status: str | None
    published_at: datetime | None
    published_url: str | None
    primary_pillar: IntegrationPillarReference | None = None
    secondary_pillars: list[IntegrationPillarReference] = Field(default_factory=list)
    pillar_assignment_confidence: float | None = None
    created_at: datetime
    updated_at: datetime


class IntegrationArticleListResponse(BaseModel):
    """Paginated lightweight article list for integrations."""

    items: list[IntegrationArticleSummaryResponse]
    total: int
    page: int
    page_size: int


class IntegrationPillarListResponse(BaseModel):
    """Pillar listing response for integrations."""

    items: list[IntegrationPillarResponse]
    total: int


class IntegrationArticlePublicationPatchRequest(BaseModel):
    """Request body for publication metadata callback updates."""

    publish_status: str | None = None
    published_at: datetime | None = None
    published_url: str | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "IntegrationArticlePublicationPatchRequest":
        if (
            self.publish_status is None
            and self.published_at is None
            and self.published_url is None
        ):
            raise ValueError("At least one publication field must be provided")
        if self.publish_status not in {None, "scheduled", "published", "failed"}:
            raise ValueError("publish_status must be one of: scheduled, published, failed")
        if self.publish_status == "published":
            if self.published_at is None or not self.published_url:
                raise ValueError(
                    "published_at and published_url are required when publish_status is published"
                )
        if self.published_url is not None:
            parsed = urlparse(self.published_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("published_url must be a valid http/https URL")
        return self


class IntegrationArticlePublicationResponse(BaseModel):
    """Integration API response for publication-state updates."""

    article_id: str
    project_id: str
    publish_status: str | None
    published_at: datetime | None
    published_url: str | None
    updated_at: datetime
