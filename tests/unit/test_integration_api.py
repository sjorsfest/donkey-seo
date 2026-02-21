"""Unit tests for the external integration API."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from app.api.integration import integration_app
from app.config import settings
from app.core.database import get_session
from app.main import create_app


class _FakeExecuteResult:
    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar_one_or_none(self) -> Any:
        return self._value


class _FakeSession:
    def __init__(self, *, article: Any, article_version: Any) -> None:
        self._article = article
        self._article_version = article_version
        self._calls = 0

    async def execute(self, _query: Any) -> _FakeExecuteResult:
        self._calls += 1
        if self._calls == 1:
            return _FakeExecuteResult(self._article)
        return _FakeExecuteResult(self._article_version)


def test_integration_docs_and_guide_are_public() -> None:
    with TestClient(create_app()) as client:
        docs_response = client.get("/api/integration/docs")
        openapi_response = client.get("/api/integration/openapi.json")
        guide_response = client.get("/api/integration/guide/donkey-client")

    assert docs_response.status_code == 200
    assert openapi_response.status_code == 200
    assert "/article/{article_id}" in openapi_response.json()["paths"]
    assert guide_response.status_code == 200
    guide_payload = guide_response.json()
    assert "modular_document" in guide_payload["markdown"]
    assert guide_payload["modular_document_contract"]["schema_version"] == "1.0"


def test_integration_article_requires_api_key() -> None:
    original_keys = settings.integration_api_keys
    settings.integration_api_keys = "valid-key"

    article = SimpleNamespace(id="article_1", project_id="project_1", current_version=1)
    now = datetime.now(timezone.utc)
    article_version = SimpleNamespace(
        id="version_1",
        article_id="article_1",
        version_number=1,
        title="Title",
        slug="title",
        primary_keyword="keyword",
        modular_document={},
        rendered_html="<article></article>",
        qa_report=None,
        status="draft",
        change_reason=None,
        generation_model="test-model",
        generation_temperature=0.4,
        created_by_regeneration=False,
        created_at=now,
        updated_at=now,
    )
    fake_session = _FakeSession(article=article, article_version=article_version)

    async def _fake_get_session() -> AsyncGenerator[_FakeSession, None]:
        yield fake_session

    integration_app.dependency_overrides[get_session] = _fake_get_session
    try:
        with TestClient(create_app()) as client:
            response = client.get(
                "/api/integration/article/article_1",
                params={"project_id": "project_1"},
            )
    finally:
        integration_app.dependency_overrides.clear()
        settings.integration_api_keys = original_keys

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_integration_article_returns_latest_version() -> None:
    original_keys = settings.integration_api_keys
    settings.integration_api_keys = "valid-key"

    article = SimpleNamespace(id="article_1", project_id="project_1", current_version=3)
    now = datetime.now(timezone.utc)
    article_version = SimpleNamespace(
        id="version_3",
        article_id="article_1",
        version_number=3,
        title="Article title",
        slug="article-title",
        primary_keyword="primary keyword",
        modular_document={
            "schema_version": "1.0",
            "seo_meta": {
                "h1": "Article title",
                "meta_title": "Article title",
                "meta_description": "Description",
                "slug": "article-title",
                "primary_keyword": "primary keyword",
            },
            "conversion_plan": {"primary_intent": "informational", "cta_strategy": []},
            "blocks": [],
        },
        rendered_html="<article><header><h1>Article title</h1></header></article>",
        qa_report={"passed": True},
        status="draft",
        change_reason="manual_regeneration",
        generation_model="openrouter:model",
        generation_temperature=0.2,
        created_by_regeneration=True,
        created_at=now,
        updated_at=now,
    )
    fake_session = _FakeSession(article=article, article_version=article_version)

    async def _fake_get_session() -> AsyncGenerator[_FakeSession, None]:
        yield fake_session

    integration_app.dependency_overrides[get_session] = _fake_get_session
    try:
        with TestClient(create_app()) as client:
            response = client.get(
                "/api/integration/article/article_1",
                params={"project_id": "project_1"},
                headers={"X-API-Key": "valid-key"},
            )
    finally:
        integration_app.dependency_overrides.clear()
        settings.integration_api_keys = original_keys

    assert response.status_code == 200
    payload = response.json()
    assert payload["article_id"] == "article_1"
    assert payload["project_id"] == "project_1"
    assert payload["version_number"] == 3
    assert payload["modular_document"]["schema_version"] == "1.0"
