"""Unit tests for content calendar state resolution."""

from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

from app.api.v1.content.routes import _resolve_calendar_state


def test_calendar_state_prefers_published_at() -> None:
    article = SimpleNamespace(
        status="needs_review",
        publish_status=None,
        published_at=datetime.now(timezone.utc),
    )

    state = _resolve_calendar_state(
        has_writer_instructions=True,
        article=article,
        publication_date=date(2026, 3, 3),
    )

    assert state == "published"


def test_calendar_state_prefers_publish_status() -> None:
    article = SimpleNamespace(
        status="draft",
        publish_status="published",
        published_at=None,
    )

    state = _resolve_calendar_state(
        has_writer_instructions=True,
        article=article,
        publication_date=date(2026, 3, 3),
    )

    assert state == "published"


def test_calendar_state_needs_review_is_article_ready() -> None:
    article = SimpleNamespace(
        status="needs_review",
        publish_status=None,
        published_at=None,
    )

    state = _resolve_calendar_state(
        has_writer_instructions=True,
        article=article,
        publication_date=date(2026, 3, 3),
        today=date(2026, 3, 3),
    )

    assert state == "article_ready"


def test_calendar_state_overdue_article_is_publish_pending() -> None:
    article = SimpleNamespace(
        status="draft",
        publish_status=None,
        published_at=None,
    )

    state = _resolve_calendar_state(
        has_writer_instructions=True,
        article=article,
        publication_date=date(2026, 3, 2),
        today=date(2026, 3, 3),
    )

    assert state == "publish_pending"


def test_calendar_state_article_ready() -> None:
    article = SimpleNamespace(
        status="draft",
        publish_status=None,
        published_at=None,
    )

    state = _resolve_calendar_state(
        has_writer_instructions=True,
        article=article,
        publication_date=date(2026, 3, 4),
        today=date(2026, 3, 3),
    )

    assert state == "article_ready"


def test_calendar_state_publication_sent() -> None:
    article = SimpleNamespace(
        status="draft",
        publish_status="publication_sent",
        published_at=None,
    )

    state = _resolve_calendar_state(
        has_writer_instructions=True,
        article=article,
        publication_date=date(2026, 3, 4),
        today=date(2026, 3, 4),
    )

    assert state == "publication_sent"


def test_calendar_state_status_published_is_published() -> None:
    article = SimpleNamespace(
        status="published",
        publish_status=None,
        published_at=None,
    )

    state = _resolve_calendar_state(
        has_writer_instructions=True,
        article=article,
        publication_date=date(2026, 3, 4),
    )

    assert state == "published"


def test_calendar_state_writer_instructions_ready() -> None:
    state = _resolve_calendar_state(
        has_writer_instructions=True,
        article=None,
        publication_date=date(2026, 3, 3),
    )

    assert state == "writer_instructions_ready"


def test_calendar_state_brief_ready() -> None:
    state = _resolve_calendar_state(
        has_writer_instructions=False,
        article=None,
        publication_date=date(2026, 3, 3),
    )

    assert state == "brief_ready"
