"""Unit tests for content start quota guards in pipeline routes."""

from __future__ import annotations

import pytest
from fastapi import HTTPException, status

from app.api.v1.pipeline.constants import CONTENT_ARTICLE_LIMIT_REACHED_DETAIL
from app.api.v1.pipeline.routes import _resolve_content_start_max_briefs


def test_resolve_content_start_max_briefs_raises_when_no_remaining_slots() -> None:
    with pytest.raises(HTTPException) as exc_info:
        _resolve_content_start_max_briefs(
            requested_max_briefs=5,
            remaining_article_slots=0,
        )

    assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
    assert exc_info.value.detail == CONTENT_ARTICLE_LIMIT_REACHED_DETAIL


def test_resolve_content_start_max_briefs_clamps_to_remaining_slots() -> None:
    effective = _resolve_content_start_max_briefs(
        requested_max_briefs=20,
        remaining_article_slots=3,
    )
    assert effective == 3
