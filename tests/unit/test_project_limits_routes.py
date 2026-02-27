"""Unit tests for project subscription limit route guards."""

from __future__ import annotations

import pytest
from fastapi import HTTPException, status

from app.api.v1.projects.constants import PROJECT_LIMIT_REACHED_DETAIL
from app.api.v1.projects.routes import _assert_project_creation_allowed
from app.services.subscription_limits import SubscriptionUsageSnapshot


@pytest.mark.asyncio
async def test_assert_project_creation_allowed_raises_when_limit_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_usage(**_kwargs: object) -> SubscriptionUsageSnapshot:
        return SubscriptionUsageSnapshot(
            plan=None,
            article_limit=3,
            project_limit=1,
            used_articles=0,
            reserved_article_slots=0,
            remaining_article_slots=3,
            remaining_article_write_slots=3,
            used_projects=1,
            remaining_project_slots=0,
        )

    monkeypatch.setattr(
        "app.api.v1.projects.routes.resolve_subscription_usage_for_user",
        _fake_usage,
    )

    with pytest.raises(HTTPException) as exc_info:
        await _assert_project_creation_allowed(
            session=object(),  # type: ignore[arg-type]
            user_id="user-1",
            subscription_plan=None,
        )

    assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
    assert PROJECT_LIMIT_REACHED_DETAIL in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_assert_project_creation_allowed_passes_with_remaining_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_usage(**_kwargs: object) -> SubscriptionUsageSnapshot:
        return SubscriptionUsageSnapshot(
            plan="growth",
            article_limit=100,
            project_limit=3,
            used_articles=10,
            reserved_article_slots=2,
            remaining_article_slots=88,
            remaining_article_write_slots=90,
            used_projects=1,
            remaining_project_slots=2,
        )

    monkeypatch.setattr(
        "app.api.v1.projects.routes.resolve_subscription_usage_for_user",
        _fake_usage,
    )

    await _assert_project_creation_allowed(
        session=object(),  # type: ignore[arg-type]
        user_id="user-1",
        subscription_plan="growth",
    )
