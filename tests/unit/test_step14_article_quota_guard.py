"""Unit tests for Step 14 subscription quota guardrails."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.steps.content.step_14_article_writer import (
    ARTICLE_QUOTA_EXHAUSTED_ERROR,
    ArticleWriterInput,
    Step14ArticleWriterService,
)
from app.services.subscription_limits import SubscriptionUsageSnapshot


class _NestedTransaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(
        self,
        _exc_type: object,
        _exc: object,
        _tb: object,
    ) -> bool:
        return False


class _FakeSession:
    def begin_nested(self) -> _NestedTransaction:
        return _NestedTransaction()

    async def flush(self) -> None:
        return None


def _fake_brief(brief_id: str) -> SimpleNamespace:
    return SimpleNamespace(id=brief_id, proposed_publication_date=None)


def _fake_featured_image(title: str) -> SimpleNamespace:
    return SimpleNamespace(status="ready", object_key=f"key-{title}", title_text=title)


def _build_service() -> Step14ArticleWriterService:
    execution = SimpleNamespace(pipeline_run_id="run-1")
    service = Step14ArticleWriterService(
        session=_FakeSession(),  # type: ignore[arg-type]
        project_id="project-1",
        execution=execution,  # type: ignore[arg-type]
    )
    service._update_progress = AsyncMock()  # type: ignore[method-assign]
    service._load_project = AsyncMock(return_value=SimpleNamespace(domain="example.com"))  # type: ignore[method-assign]
    service._load_brand = AsyncMock(return_value=None)  # type: ignore[method-assign]
    service.get_run_strategy = AsyncMock(return_value=SimpleNamespace(conversion_intents=[]))  # type: ignore[method-assign]
    service._load_pipeline_run = AsyncMock(return_value=SimpleNamespace(id="run-1"))  # type: ignore[method-assign]
    service._build_brand_context = lambda _brand: ""  # type: ignore[method-assign]
    service._brief_payload = lambda brief, *, locked_title=None: {  # type: ignore[method-assign]
        "id": str(brief.id),
        "locked_title": locked_title or "",
    }
    return service


@pytest.mark.asyncio
async def test_step14_raises_when_quota_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _build_service()
    brief = _fake_brief("brief-1")
    service._load_briefs = AsyncMock(return_value=[brief])  # type: ignore[method-assign]
    service._load_writer_instructions = AsyncMock(return_value={"brief-1": {}})  # type: ignore[method-assign]
    service._load_brief_deltas = AsyncMock(return_value={})  # type: ignore[method-assign]
    service._load_featured_images = AsyncMock(  # type: ignore[method-assign]
        return_value={"brief-1": _fake_featured_image("Locked title")}
    )
    service._load_existing_article_brief_ids = AsyncMock(return_value=set())  # type: ignore[method-assign]
    service._load_project_authors = AsyncMock(return_value=[])  # type: ignore[method-assign]

    async def _usage(**_kwargs: object) -> SubscriptionUsageSnapshot:
        return SubscriptionUsageSnapshot(
            plan=None,
            article_limit=3,
            project_limit=1,
            used_articles=3,
            reserved_article_slots=0,
            remaining_article_slots=0,
            remaining_article_write_slots=0,
            used_projects=1,
            remaining_project_slots=0,
        )

    monkeypatch.setattr(
        "app.services.steps.content.step_14_article_writer.resolve_subscription_usage_for_project",
        _usage,
    )

    with pytest.raises(ValueError, match=ARTICLE_QUOTA_EXHAUSTED_ERROR):
        await service._execute(ArticleWriterInput(project_id="project-1"))


@pytest.mark.asyncio
async def test_step14_persists_only_up_to_quota_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _build_service()
    briefs = [_fake_brief("brief-1"), _fake_brief("brief-2")]
    service._load_briefs = AsyncMock(return_value=briefs)  # type: ignore[method-assign]
    service._load_writer_instructions = AsyncMock(  # type: ignore[method-assign]
        return_value={"brief-1": {}, "brief-2": {}}
    )
    service._load_brief_deltas = AsyncMock(return_value={})  # type: ignore[method-assign]
    service._load_featured_images = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "brief-1": _fake_featured_image("Title One"),
            "brief-2": _fake_featured_image("Title Two"),
        }
    )
    service._load_existing_article_brief_ids = AsyncMock(return_value=set())  # type: ignore[method-assign]
    service._load_project_authors = AsyncMock(return_value=[])  # type: ignore[method-assign]

    usage_calls = 0

    async def _usage(**_kwargs: object) -> SubscriptionUsageSnapshot:
        nonlocal usage_calls
        usage_calls += 1
        return SubscriptionUsageSnapshot(
            plan="starter",
            article_limit=30,
            project_limit=1,
            used_articles=29,
            reserved_article_slots=0,
            remaining_article_slots=1,
            remaining_article_write_slots=1,
            used_projects=1,
            remaining_project_slots=0,
        )

    monkeypatch.setattr(
        "app.services.steps.content.step_14_article_writer.resolve_subscription_usage_for_project",
        _usage,
    )

    class _FakeGenerator:
        def __init__(self, _domain: str) -> None:
            pass

        async def generate_with_repair(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                title="Generated",
                slug="generated",
                primary_keyword="kw",
                modular_document={"blocks": [], "seo_meta": {}},
                rendered_html="<article></article>",
                qa_report={},
                status="draft",
                generation_model="model",
                generation_temperature=0.2,
            )

    monkeypatch.setattr(
        "app.services.steps.content.step_14_article_writer.ArticleGenerationService",
        _FakeGenerator,
    )
    monkeypatch.setattr(
        "app.services.steps.content.step_14_article_writer.modular_featured_image_payload",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        "app.services.steps.content.step_14_article_writer.render_modular_document",
        lambda _document: "<article>rendered</article>",
    )
    monkeypatch.setattr(
        "app.services.steps.content.step_14_article_writer.schedule_publication_webhook_for_article",
        AsyncMock(),
    )

    created_articles: list[str] = []

    def _create_article(_session: object, dto: object) -> SimpleNamespace:
        created_articles.append(str(dto.brief_id))  # type: ignore[attr-defined]
        return SimpleNamespace(id=f"article-{len(created_articles)}")

    monkeypatch.setattr(
        "app.services.steps.content.step_14_article_writer.ContentArticle.create",
        _create_article,
    )
    monkeypatch.setattr(
        "app.services.steps.content.step_14_article_writer.ContentArticleVersion.create",
        lambda _session, _dto: None,
    )

    output = await service._execute(ArticleWriterInput(project_id="project-1"))

    assert usage_calls >= 2
    assert created_articles == ["brief-1"]
    assert output.articles_generated == 1
    assert output.articles_failed == 1
    assert any(item["reason"] == "article_quota_reached" for item in output.failures)
