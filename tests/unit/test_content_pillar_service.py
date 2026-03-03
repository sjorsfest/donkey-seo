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

    assert len(names) >= service.MIN_BOOTSTRAP_PILLARS
    assert len(names) <= service.MAX_ACTIVE_PILLARS
    assert "Blog" in names
    assert set(names).issubset({"Blog", "Comparison", "How To", "Use Case", "Commercial"})


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


def test_assign_docs_uses_high_level_taxonomy_labels() -> None:
    service = _service()
    docs = [
        TopicDoc(
            topic_id="topic-comparison",
            text="Zendesk vs Intercom alternatives",
            tokens=["zendesk", "intercom", "alternatives"],
        ),
        TopicDoc(
            topic_id="topic-howto",
            text="How to set up support automations",
            tokens=["how", "set", "support", "automations"],
        ),
        TopicDoc(
            topic_id="topic-blog",
            text="Customer support trends for 2026",
            tokens=["customer", "support", "trends", "2026"],
        ),
    ]
    pillars = [
        SimpleNamespace(id="pillar-blog", name="Blog", description=None),
        SimpleNamespace(id="pillar-comparison", name="Comparison", description=None),
        SimpleNamespace(id="pillar-howto", name="How To", description=None),
    ]

    assignments = service._assign_docs(docs, pillars)

    assert assignments["topic-comparison"].primary_pillar_id == "pillar-comparison"
    assert assignments["topic-comparison"].assignment_method == "auto_high_level"
    assert assignments["topic-howto"].primary_pillar_id == "pillar-howto"
    assert assignments["topic-howto"].assignment_method == "auto_high_level"
    assert assignments["topic-blog"].primary_pillar_id == "pillar-blog"
    assert assignments["topic-blog"].assignment_method == "auto_high_level"


def test_derive_bootstrap_names_adds_extra_high_level_archetypes_when_signaled() -> None:
    service = _service()
    docs = [
        TopicDoc(topic_id="t1", text="Pricing and demo for support platform", tokens=["pricing", "demo"]),
        TopicDoc(topic_id="t2", text="Workflow use case for agencies", tokens=["workflow", "agencies"]),
    ]

    names = service._derive_bootstrap_names(docs)

    assert "Blog" in names
    assert "Commercial" in names
    assert "Use Case" in names
