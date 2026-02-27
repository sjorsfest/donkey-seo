"""Unit tests for orchestrator subscription cap enforcement."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.schemas.pipeline import ContentPipelineConfig
from app.services.pipeline_orchestrator import PipelineOrchestrator
from app.services.subscription_limits import SubscriptionUsageSnapshot


class _ExistingResult:
    def __init__(self, topic_ids: list[str]) -> None:
        self._topic_ids = topic_ids

    def scalars(self) -> list[str]:
        return self._topic_ids


class _SessionWithExistingDispatch:
    def __init__(self, topic_ids: list[str]) -> None:
        self._topic_ids = topic_ids
        self.commit = AsyncMock()

    async def execute(self, _query: object) -> _ExistingResult:
        return _ExistingResult(self._topic_ids)


@pytest.mark.asyncio
async def test_dispatch_new_topics_truncates_to_remaining_article_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = PipelineOrchestrator(None, "project-1")
    orchestrator.task_manager = SimpleNamespace(set_task_state=AsyncMock())
    run = SimpleNamespace(id="discovery-1", steps_config={})

    session = _SessionWithExistingDispatch([])

    @asynccontextmanager
    async def _active_session():
        yield session

    orchestrator._active_session = lambda: _active_session()  # type: ignore[assignment]

    async def _fake_usage(**_kwargs: object) -> SubscriptionUsageSnapshot:
        return SubscriptionUsageSnapshot(
            plan=None,
            article_limit=3,
            project_limit=1,
            used_articles=1,
            reserved_article_slots=0,
            remaining_article_slots=2,
            remaining_article_write_slots=2,
            used_projects=1,
            remaining_project_slots=0,
        )

    monkeypatch.setattr(
        "app.services.pipeline_orchestrator.resolve_subscription_usage_for_project",
        _fake_usage,
    )

    created_topic_ids: list[str] = []

    def _fake_create(_session: object, dto: object) -> SimpleNamespace:
        topic_id = str(dto.source_topic_id)  # type: ignore[attr-defined]
        created_topic_ids.append(topic_id)
        return SimpleNamespace(id=f"content-{len(created_topic_ids)}", source_topic_id=topic_id)

    monkeypatch.setattr("app.services.pipeline_orchestrator.PipelineRun.create", _fake_create)

    queue = SimpleNamespace(enqueue_start=AsyncMock())
    monkeypatch.setattr(
        "app.services.pipeline_orchestrator.get_content_pipeline_task_manager",
        lambda: queue,
    )

    await orchestrator._dispatch_new_topics_from_discovery(
        run=run,  # type: ignore[arg-type]
        accepted_topic_ids=["topic-1", "topic-2", "topic-3"],
        content_config=ContentPipelineConfig(),
    )

    assert created_topic_ids == ["topic-1", "topic-2"]
    assert queue.enqueue_start.await_count == 2


@pytest.mark.asyncio
async def test_dispatch_new_topics_skips_when_no_remaining_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = PipelineOrchestrator(None, "project-1")
    orchestrator.task_manager = SimpleNamespace(set_task_state=AsyncMock())
    run = SimpleNamespace(id="discovery-1", steps_config={})

    session = _SessionWithExistingDispatch([])

    @asynccontextmanager
    async def _active_session():
        yield session

    orchestrator._active_session = lambda: _active_session()  # type: ignore[assignment]

    async def _fake_usage(**_kwargs: object) -> SubscriptionUsageSnapshot:
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
        "app.services.pipeline_orchestrator.resolve_subscription_usage_for_project",
        _fake_usage,
    )

    create_mock = Mock(return_value=SimpleNamespace(id="should-not-happen"))
    monkeypatch.setattr("app.services.pipeline_orchestrator.PipelineRun.create", create_mock)

    queue = SimpleNamespace(enqueue_start=AsyncMock())
    monkeypatch.setattr(
        "app.services.pipeline_orchestrator.get_content_pipeline_task_manager",
        lambda: queue,
    )

    await orchestrator._dispatch_new_topics_from_discovery(
        run=run,  # type: ignore[arg-type]
        accepted_topic_ids=["topic-1", "topic-2"],
        content_config=ContentPipelineConfig(),
    )

    assert create_mock.call_count == 0
    queue.enqueue_start.assert_not_awaited()
