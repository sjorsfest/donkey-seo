"""Unit tests for content calendar state resolution."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from app.api.v1.content.routes import _resolve_calendar_state


def test_calendar_state_prefers_published_at() -> None:
    article = SimpleNamespace(
        status="needs_review",
        publish_status=None,
        published_at=datetime.now(timezone.utc),
    )

    state = _resolve_calendar_state(has_writer_instructions=True, article=article)

    assert state == "published"


def test_calendar_state_prefers_publish_status() -> None:
    article = SimpleNamespace(
        status="draft",
        publish_status="published",
        published_at=None,
    )

    state = _resolve_calendar_state(has_writer_instructions=True, article=article)

    assert state == "published"


def test_calendar_state_needs_review() -> None:
    article = SimpleNamespace(
        status="needs_review",
        publish_status=None,
        published_at=None,
    )

    state = _resolve_calendar_state(has_writer_instructions=True, article=article)

    assert state == "article_needs_review"


def test_calendar_state_article_ready() -> None:
    article = SimpleNamespace(
        status="draft",
        publish_status=None,
        published_at=None,
    )

    state = _resolve_calendar_state(has_writer_instructions=True, article=article)

    assert state == "article_ready"


def test_calendar_state_writer_instructions_ready() -> None:
    state = _resolve_calendar_state(has_writer_instructions=True, article=None)

    assert state == "writer_instructions_ready"


def test_calendar_state_brief_ready() -> None:
    state = _resolve_calendar_state(has_writer_instructions=False, article=None)

    assert state == "brief_ready"
