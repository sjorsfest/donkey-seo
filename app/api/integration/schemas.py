"""Pydantic schemas for external integration API responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class IntegrationIndexResponse(BaseModel):
    """Top-level integration API index."""

    service: str
    docs_path: str
    openapi_path: str
    guide_path: str
    guide_markdown_path: str
    article_latest_path_template: str
    article_version_path_template: str
    auth_header: str


class IntegrationGuideResponse(BaseModel):
    """Detailed implementation guide payload."""

    title: str
    schema_version: str
    markdown: str
    modular_document_contract: dict[str, Any]


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
