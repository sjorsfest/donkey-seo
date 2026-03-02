"""Unit tests for content pillar taxonomy logic."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.content_pillar_service import ContentPillarService, TopicDoc


class _FakeSession:
    pass


def _service() -> ContentPillarService:
    return ContentPillarService(cast(AsyncSession, _FakeSession()))


def test_derive_bootstrap_names_is_bounded() -> None:
    service = _service()
    docs = [
        TopicDoc(topic_id="t1", text="Customer support automation", tokens=["customer", "support", "automation"]),
        TopicDoc(topic_id="t2", text="Support workflows and automations", tokens=["support", "workflow", "automation"]),
        TopicDoc(topic_id="t3", text="Ticket routing automation", tokens=["ticket", "routing", "automation"]),
        TopicDoc(topic_id="t4", text="Knowledge base support", tokens=["knowledge", "base", "support"]),
    ]

    names = service._derive_bootstrap_names(docs)

    assert names
    assert len(names) <= service.MAX_ACTIVE_PILLARS
    assert len(names) >= service.MIN_BOOTSTRAP_PILLARS


def test_assign_docs_produces_primary_and_optional_secondary() -> None:
    service = _service()
    docs = [
        TopicDoc(topic_id="topic-1", text="support automation", tokens=["support", "automation"]),
        TopicDoc(topic_id="topic-2", text="headshot generator", tokens=["headshot", "generator"]),
    ]
    pillars = [
        SimpleNamespace(id="pillar-support", name="Support", description="Support workflows"),
        SimpleNamespace(id="pillar-automation", name="Automation", description="Automation guides"),
        SimpleNamespace(id="pillar-headshot", name="Headshot", description="Headshot content"),
    ]

    assignments = service._assign_docs(docs, pillars)

    first = assignments["topic-1"]
    assert first.primary_pillar_id in {"pillar-support", "pillar-automation"}
    assert first.assignment_method == "auto"

    second = assignments["topic-2"]
    assert second.primary_pillar_id == "pillar-headshot"
    assert second.assignment_method == "auto"


def test_assign_docs_uses_forced_fallback_when_no_match() -> None:
    service = _service()
    docs = [
        TopicDoc(topic_id="topic-1", text="very niche topic", tokens=["niche", "topic"]),
    ]
    pillars = [
        SimpleNamespace(id="pillar-support", name="Support", description="Support workflows"),
    ]

    assignments = service._assign_docs(docs, pillars)

    assignment = assignments["topic-1"]
    assert assignment.primary_pillar_id == "pillar-support"
    assert assignment.assignment_method == "auto_forced"
