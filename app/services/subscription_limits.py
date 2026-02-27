"""Subscription usage and hard-cap helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.content import ContentArticle
from app.models.pipeline import PipelineRun
from app.models.project import Project
from app.models.user import User
from app.services.billing import (
    PlanKey,
    normalize_plan,
    resolve_article_limit,
    resolve_project_limit,
    resolve_usage_window,
)

RESERVED_CONTENT_RUN_STATUSES: tuple[str, ...] = ("pending", "running", "paused")
DEFAULT_MANUAL_CONTENT_MAX_BRIEFS = 20


@dataclass(frozen=True)
class SubscriptionUsageSnapshot:
    """Usage/capacity snapshot for one user subscription state."""

    plan: PlanKey | None
    article_limit: int
    project_limit: int
    used_articles: int
    reserved_article_slots: int
    remaining_article_slots: int
    remaining_article_write_slots: int
    used_projects: int
    remaining_project_slots: int


def coerce_positive_int(value: Any, *, default: int) -> int:
    """Parse positive integer with fallback."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def reserved_slots_for_content_run(
    *,
    source_topic_id: str | None,
    steps_config: Any,
    default_manual_max_briefs: int = DEFAULT_MANUAL_CONTENT_MAX_BRIEFS,
) -> int:
    """Resolve reserved article slots for one in-flight content run."""
    if source_topic_id:
        return 1

    max_briefs_value: Any = None
    if isinstance(steps_config, dict):
        step_inputs = steps_config.get("step_inputs")
        if isinstance(step_inputs, dict):
            step1_payload = step_inputs.get("1")
            if step1_payload is None:
                step1_payload = step_inputs.get(1)
            if isinstance(step1_payload, dict):
                max_briefs_value = step1_payload.get("max_briefs")
        if max_briefs_value is None:
            content_config = steps_config.get("content")
            if isinstance(content_config, dict):
                max_briefs_value = content_config.get("max_briefs")

    return coerce_positive_int(max_briefs_value, default=default_manual_max_briefs)


async def resolve_subscription_usage_for_user(
    *,
    session: AsyncSession,
    user_id: str,
    subscription_plan: str | None,
    exclude_run_id: str | None = None,
) -> SubscriptionUsageSnapshot:
    """Resolve current usage and remaining capacity for one user."""
    plan = normalize_plan(subscription_plan)
    usage_window = resolve_usage_window(plan)
    article_limit = resolve_article_limit(plan)
    project_limit = resolve_project_limit(plan)

    used_projects_query = (
        select(func.count())
        .select_from(Project)
        .where(Project.user_id == user_id)
    )
    used_projects = int(await session.scalar(used_projects_query) or 0)

    used_articles_query = (
        select(func.count())
        .select_from(ContentArticle)
        .join(Project, ContentArticle.project_id == Project.id)
        .where(Project.user_id == user_id)
    )
    if usage_window.period_start is not None:
        used_articles_query = used_articles_query.where(
            ContentArticle.generated_at >= usage_window.period_start
        )
    if usage_window.period_end is not None:
        used_articles_query = used_articles_query.where(
            ContentArticle.generated_at < usage_window.period_end
        )
    used_articles = int(await session.scalar(used_articles_query) or 0)

    reserved_query = (
        select(PipelineRun.source_topic_id, PipelineRun.steps_config)
        .select_from(PipelineRun)
        .join(Project, PipelineRun.project_id == Project.id)
        .where(
            Project.user_id == user_id,
            PipelineRun.pipeline_module == "content",
            PipelineRun.status.in_(RESERVED_CONTENT_RUN_STATUSES),
        )
    )
    if exclude_run_id:
        reserved_query = reserved_query.where(PipelineRun.id != exclude_run_id)

    reserved_rows = (await session.execute(reserved_query)).all()
    reserved_article_slots = sum(
        reserved_slots_for_content_run(
            source_topic_id=(str(source_topic_id) if source_topic_id is not None else None),
            steps_config=steps_config,
        )
        for source_topic_id, steps_config in reserved_rows
    )

    remaining_article_slots = max(article_limit - used_articles - reserved_article_slots, 0)
    remaining_article_write_slots = max(article_limit - used_articles, 0)
    remaining_project_slots = max(project_limit - used_projects, 0)

    return SubscriptionUsageSnapshot(
        plan=plan,
        article_limit=article_limit,
        project_limit=project_limit,
        used_articles=used_articles,
        reserved_article_slots=reserved_article_slots,
        remaining_article_slots=remaining_article_slots,
        remaining_article_write_slots=remaining_article_write_slots,
        used_projects=used_projects,
        remaining_project_slots=remaining_project_slots,
    )


async def resolve_subscription_usage_for_project(
    *,
    session: AsyncSession,
    project_id: str,
    exclude_run_id: str | None = None,
) -> SubscriptionUsageSnapshot:
    """Resolve usage and capacity by loading owner from project id."""
    owner_result = await session.execute(
        select(Project.user_id, User.subscription_plan)
        .join(User, Project.user_id == User.id)
        .where(Project.id == project_id)
        .limit(1)
    )
    owner = owner_result.one_or_none()
    if owner is None:
        raise ValueError(f"Project not found: {project_id}")

    user_id, subscription_plan = owner
    return await resolve_subscription_usage_for_user(
        session=session,
        user_id=str(user_id),
        subscription_plan=(str(subscription_plan) if subscription_plan is not None else None),
        exclude_run_id=exclude_run_id,
    )
