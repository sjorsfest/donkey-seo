"""Unit tests for subscription usage/capacity helpers."""

from __future__ import annotations

import pytest

from app.services.subscription_limits import (
    resolve_subscription_usage_for_project,
    resolve_subscription_usage_for_user,
    reserved_slots_for_content_run,
)


class _FakeResult:
    def __init__(
        self,
        *,
        rows: list[tuple[object, object]] | None = None,
        one: tuple[object, object] | None = None,
    ) -> None:
        self._rows = rows or []
        self._one = one

    def all(self) -> list[tuple[object, object]]:
        return self._rows

    def one_or_none(self) -> tuple[object, object] | None:
        return self._one


class _FakeSession:
    def __init__(
        self,
        *,
        scalar_values: list[int],
        execute_results: list[_FakeResult],
    ) -> None:
        self._scalar_values = iter(scalar_values)
        self._execute_results = iter(execute_results)

    async def scalar(self, _query: object) -> int:
        return next(self._scalar_values)

    async def execute(self, _query: object) -> _FakeResult:
        return next(self._execute_results)


def test_reserved_slots_for_content_run_source_topic_is_one() -> None:
    assert reserved_slots_for_content_run(source_topic_id="topic-1", steps_config={}) == 1


def test_reserved_slots_for_content_run_manual_max_briefs_parsing() -> None:
    slots = reserved_slots_for_content_run(
        source_topic_id=None,
        steps_config={"step_inputs": {"1": {"max_briefs": 4}}},
    )
    assert slots == 4

    slots_from_content = reserved_slots_for_content_run(
        source_topic_id=None,
        steps_config={"content": {"max_briefs": 6}},
    )
    assert slots_from_content == 6

    slots_default = reserved_slots_for_content_run(source_topic_id=None, steps_config={})
    assert slots_default == 20


@pytest.mark.asyncio
async def test_resolve_subscription_usage_for_user_includes_reserved_slots() -> None:
    session = _FakeSession(
        scalar_values=[1, 1],
        execute_results=[
            _FakeResult(
                rows=[
                    ("topic-1", {}),
                    (None, {"step_inputs": {"1": {"max_briefs": 2}}}),
                ]
            )
        ],
    )

    usage = await resolve_subscription_usage_for_user(
        session=session,  # type: ignore[arg-type]
        user_id="user-1",
        subscription_plan=None,
    )

    assert usage.article_limit == 3
    assert usage.project_limit == 1
    assert usage.used_projects == 1
    assert usage.used_articles == 1
    assert usage.reserved_article_slots == 3
    assert usage.remaining_article_slots == 0
    assert usage.remaining_article_write_slots == 2
    assert usage.remaining_project_slots == 0


@pytest.mark.asyncio
async def test_resolve_subscription_usage_for_project_loads_owner_and_plan() -> None:
    session = _FakeSession(
        scalar_values=[2, 10],
        execute_results=[
            _FakeResult(one=("user-1", "growth")),
            _FakeResult(rows=[(None, {"content": {"max_briefs": 5}})]),
        ],
    )

    usage = await resolve_subscription_usage_for_project(
        session=session,  # type: ignore[arg-type]
        project_id="project-1",
    )

    assert usage.plan == "growth"
    assert usage.article_limit == 100
    assert usage.project_limit == 3
    assert usage.used_projects == 2
    assert usage.used_articles == 10
    assert usage.reserved_article_slots == 5
    assert usage.remaining_article_slots == 85
    assert usage.remaining_project_slots == 1
