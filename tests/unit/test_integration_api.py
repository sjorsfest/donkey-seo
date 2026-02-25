"""Unit tests for the external integration API."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.api.integration import integration_app
from app.config import settings
from app.core.database import get_session
from app.main import create_app

INTEGRATION_API_BASE_PATH = settings.versioned_integration_api_prefix


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


class _MutableArticle:
    def __init__(
        self,
        *,
        article_id: str,
        project_id: str,
        brief_id: str = "brief_1",
        publish_status: str | None = None,
        published_at: datetime | None = None,
        published_url: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        self.id = article_id
        self.project_id = project_id
        self.brief_id = brief_id
        self.publish_status = publish_status
        self.published_at = published_at
        self.published_url = published_url
        self.updated_at = now

    def patch(self, _session: Any, dto: Any) -> "_MutableArticle":
        for key, value in dto.to_patch_dict().items():
            setattr(self, key, value)
        self.updated_at = datetime.now(timezone.utc)
        return self


class _FakePublicationSession:
    def __init__(self, *, article: _MutableArticle | None) -> None:
        self._article = article
        self.flush_called = False
        self.refresh_called = False

    async def execute(self, _query: Any) -> _FakeExecuteResult:
        return _FakeExecuteResult(self._article)

    async def flush(self) -> None:
        self.flush_called = True

    async def refresh(self, _instance: Any) -> None:
        self.refresh_called = True


def test_integration_docs_and_guide_are_public() -> None:
    with TestClient(create_app()) as client:
        docs_response = client.get(f"{INTEGRATION_API_BASE_PATH}/docs")
        openapi_response = client.get(f"{INTEGRATION_API_BASE_PATH}/openapi.json")
        guide_response = client.get(f"{INTEGRATION_API_BASE_PATH}/guide/donkey-client")

    assert docs_response.status_code == 200
    assert openapi_response.status_code == 200
    assert "/article/{article_id}" in openapi_response.json()["paths"]
    assert "/article/{article_id}/publication" in openapi_response.json()["paths"]
    assert guide_response.status_code == 200
    guide_payload = guide_response.json()
    assert "modular_document" in guide_payload["markdown"]
    assert "content.article.publish_requested" in guide_payload["markdown"]
    assert guide_payload["modular_document_contract"]["schema_version"] == "1.0"


def test_integration_index_exposes_publication_callback_template() -> None:
    with TestClient(create_app()) as client:
        response = client.get(f"{INTEGRATION_API_BASE_PATH}/")

    assert response.status_code == 200
    payload = response.json()
    assert (
        payload["article_publication_patch_path_template"]
        == "/article/{article_id}/publication?project_id={project_id}"
    )


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
                f"{INTEGRATION_API_BASE_PATH}/article/article_1",
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
                f"{INTEGRATION_API_BASE_PATH}/article/article_1",
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


def test_integration_publication_patch_requires_api_key() -> None:
    original_keys = settings.integration_api_keys
    settings.integration_api_keys = "valid-key"
    article = _MutableArticle(article_id="article_1", project_id="project_1")
    fake_session = _FakePublicationSession(article=article)

    async def _fake_get_session() -> AsyncGenerator[_FakePublicationSession, None]:
        yield fake_session

    integration_app.dependency_overrides[get_session] = _fake_get_session
    try:
        with TestClient(create_app()) as client:
            response = client.patch(
                f"{INTEGRATION_API_BASE_PATH}/article/article_1/publication",
                params={"project_id": "project_1"},
                json={"publish_status": "scheduled"},
            )
    finally:
        integration_app.dependency_overrides.clear()
        settings.integration_api_keys = original_keys

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_integration_publication_patch_updates_article_and_cancels_pending(monkeypatch: Any) -> None:
    original_keys = settings.integration_api_keys
    settings.integration_api_keys = "valid-key"
    article = _MutableArticle(article_id="article_1", project_id="project_1")
    fake_session = _FakePublicationSession(article=article)
    cancel_mock = AsyncMock(return_value=1)
    monkeypatch.setattr(
        "app.api.integration.routes.cancel_pending_publication_webhook_deliveries",
        cancel_mock,
    )
    resolve_links_mock = AsyncMock(return_value=0)
    monkeypatch.setattr(
        "app.api.integration.routes.resolve_deferred_internal_links_for_published_article",
        resolve_links_mock,
    )

    async def _fake_get_session() -> AsyncGenerator[_FakePublicationSession, None]:
        yield fake_session

    integration_app.dependency_overrides[get_session] = _fake_get_session
    try:
        with TestClient(create_app()) as client:
            response = client.patch(
                f"{INTEGRATION_API_BASE_PATH}/article/article_1/publication",
                params={"project_id": "project_1"},
                headers={"X-API-Key": "valid-key"},
                json={
                    "publish_status": "published",
                    "published_at": "2026-02-24T10:00:00Z",
                    "published_url": "https://example.com/blog/article-1",
                },
            )
    finally:
        integration_app.dependency_overrides.clear()
        settings.integration_api_keys = original_keys

    assert response.status_code == 200
    payload = response.json()
    assert payload["article_id"] == "article_1"
    assert payload["project_id"] == "project_1"
    assert payload["publish_status"] == "published"
    assert payload["published_url"] == "https://example.com/blog/article-1"
    assert fake_session.flush_called is True
    assert fake_session.refresh_called is True
    cancel_mock.assert_awaited_once()
    resolve_links_mock.assert_awaited_once()


def test_integration_publication_patch_validates_required_published_fields() -> None:
    original_keys = settings.integration_api_keys
    settings.integration_api_keys = "valid-key"
    article = _MutableArticle(article_id="article_1", project_id="project_1")
    fake_session = _FakePublicationSession(article=article)

    async def _fake_get_session() -> AsyncGenerator[_FakePublicationSession, None]:
        yield fake_session

    integration_app.dependency_overrides[get_session] = _fake_get_session
    try:
        with TestClient(create_app()) as client:
            response = client.patch(
                f"{INTEGRATION_API_BASE_PATH}/article/article_1/publication",
                params={"project_id": "project_1"},
                headers={"X-API-Key": "valid-key"},
                json={"publish_status": "published"},
            )
    finally:
        integration_app.dependency_overrides.clear()
        settings.integration_api_keys = original_keys

    assert response.status_code == 422


def test_integration_publication_patch_rejects_empty_payload() -> None:
    original_keys = settings.integration_api_keys
    settings.integration_api_keys = "valid-key"
    article = _MutableArticle(article_id="article_1", project_id="project_1")
    fake_session = _FakePublicationSession(article=article)

    async def _fake_get_session() -> AsyncGenerator[_FakePublicationSession, None]:
        yield fake_session

    integration_app.dependency_overrides[get_session] = _fake_get_session
    try:
        with TestClient(create_app()) as client:
            response = client.patch(
                f"{INTEGRATION_API_BASE_PATH}/article/article_1/publication",
                params={"project_id": "project_1"},
                headers={"X-API-Key": "valid-key"},
                json={},
            )
    finally:
        integration_app.dependency_overrides.clear()
        settings.integration_api_keys = original_keys

    assert response.status_code == 422
