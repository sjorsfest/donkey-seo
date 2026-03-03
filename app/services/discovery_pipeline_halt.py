"""Auto-halt and auto-resume guard for discovery pipeline capacity."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.database import get_session_context
from app.core.redis import get_redis_client
from app.models.content import ContentBrief
from app.models.pipeline import PipelineRun
from app.models.project import Project
from app.models.user import User
from app.services.billing import normalize_plan
from app.services.pipeline_task_manager import (
    PipelineQueueFullError,
    get_discovery_pipeline_task_manager,
)

logger = logging.getLogger(__name__)

DISCOVERY_AUTO_HALT_REASON_CODE = "discovery_auto_halt_scheduled_content_backlog"
DISCOVERY_AUTO_HALT_STAGE = "Discovery pipeline auto-halted due to scheduled content backlog"
DISCOVERY_AUTO_HALT_MAX_PROJECTS_PER_SWEEP = 200
DISCOVERY_AUTO_HALT_RECONCILIATION_METRICS_KEY = "pipeline:discovery:auto_halt:reconciliation"


@dataclass(frozen=True)
class DiscoveryPipelineHaltState:
    """Backpressure snapshot for one project's discovery pipeline."""

    is_paid_user: bool
    upcoming_scheduled_items: int
    halt_threshold: int
    window_days: int

    @property
    def should_halt(self) -> bool:
        return self.is_paid_user and self.upcoming_scheduled_items > self.halt_threshold

    @property
    def should_resume(self) -> bool:
        return self.is_paid_user and self.upcoming_scheduled_items < self.halt_threshold


def build_discovery_auto_halt_reason(state: DiscoveryPipelineHaltState) -> str:
    """Build machine-readable run pause reason for auto-halt."""
    return (
        f"{DISCOVERY_AUTO_HALT_REASON_CODE}:"
        f"scheduled_items={state.upcoming_scheduled_items};"
        f"threshold={state.halt_threshold};"
        f"window_days={state.window_days}"
    )


def build_discovery_auto_halt_detail(
    *,
    upcoming_scheduled_items: int,
    halt_threshold: int,
    window_days: int,
) -> str:
    """Build API-facing detail message for discovery auto-halt."""
    return (
        "Discovery pipeline is auto-halted because "
        f"{upcoming_scheduled_items} briefs/articles are scheduled in the next "
        f"{window_days} days (halt threshold is > {halt_threshold})."
    )


def is_discovery_auto_halt_reason(error_message: str | None) -> bool:
    """Whether a run error_message indicates the discovery auto-halt guard."""
    if not error_message:
        return False
    return error_message.startswith(f"{DISCOVERY_AUTO_HALT_REASON_CODE}:")


async def resolve_discovery_pipeline_halt_state(
    *,
    session: AsyncSession,
    project_id: str,
    as_of_date: date | None = None,
) -> DiscoveryPipelineHaltState:
    """Resolve whether discovery should halt or resume for one project."""
    halt_threshold = max(1, int(settings.discovery_pipeline_halt_threshold))
    window_days = max(1, int(settings.discovery_pipeline_halt_window_days))
    window_start = as_of_date or datetime.now(timezone.utc).date()
    window_end = window_start + timedelta(days=window_days)

    owner_result = await session.execute(
        select(Project.id, User.subscription_plan)
        .select_from(Project)
        .join(User, Project.user_id == User.id)
        .where(Project.id == project_id)
        .limit(1)
    )
    owner = owner_result.one_or_none()
    if owner is None:
        raise ValueError(f"Project not found: {project_id}")
    _, subscription_plan = owner

    normalized_plan = normalize_plan(str(subscription_plan) if subscription_plan is not None else None)
    is_paid_user = normalized_plan is not None
    scheduled_items = 0
    if is_paid_user:
        scheduled_items_query = (
            select(func.count())
            .select_from(ContentBrief)
            .where(
                ContentBrief.project_id == project_id,
                ContentBrief.proposed_publication_date.is_not(None),
                ContentBrief.proposed_publication_date >= window_start,
                ContentBrief.proposed_publication_date <= window_end,
            )
        )
        scheduled_items = int(await session.scalar(scheduled_items_query) or 0)

    return DiscoveryPipelineHaltState(
        is_paid_user=is_paid_user,
        upcoming_scheduled_items=scheduled_items,
        halt_threshold=halt_threshold,
        window_days=window_days,
    )


async def reconcile_discovery_auto_halted_runs(
    *,
    max_projects: int = DISCOVERY_AUTO_HALT_MAX_PROJECTS_PER_SWEEP,
) -> int:
    """Daily reconciliation: resume paused discovery runs once backlog is below threshold."""
    project_limit = max(1, int(max_projects))
    resumed_count = 0
    queue = get_discovery_pipeline_task_manager()

    async with get_session_context() as session:
        paused_result = await session.execute(
            select(PipelineRun.id, PipelineRun.project_id)
            .where(
                PipelineRun.pipeline_module == "discovery",
                PipelineRun.status == "paused",
                PipelineRun.error_message.like(f"{DISCOVERY_AUTO_HALT_REASON_CODE}:%"),
            )
            .order_by(PipelineRun.project_id.asc(), PipelineRun.created_at.desc())
        )
        paused_rows = list(paused_result.all())

        latest_by_project: dict[str, str] = {}
        for run_id, project_id in paused_rows:
            project_id_str = str(project_id)
            if project_id_str in latest_by_project:
                continue
            latest_by_project[project_id_str] = str(run_id)
            if len(latest_by_project) >= project_limit:
                break

        for project_id, run_id in latest_by_project.items():
            halt_state = await resolve_discovery_pipeline_halt_state(
                session=session,
                project_id=project_id,
            )
            if not halt_state.should_resume:
                continue

            running_result = await session.execute(
                select(PipelineRun.id)
                .where(
                    PipelineRun.project_id == project_id,
                    PipelineRun.pipeline_module == "discovery",
                    PipelineRun.status == "running",
                )
                .limit(1)
            )
            if running_result.scalar_one_or_none() is not None:
                continue

            try:
                await queue.enqueue_resume(
                    project_id=project_id,
                    run_id=run_id,
                )
                resumed_count += 1
            except PipelineQueueFullError:
                logger.warning(
                    "Skipping discovery auto-resume sweep enqueue because queue is full",
                    extra={
                        "project_id": project_id,
                        "run_id": run_id,
                    },
                )

    return resumed_count


async def write_discovery_reconciliation_metrics(
    *,
    started_at: datetime,
    finished_at: datetime,
    status: Literal["ok", "error"],
    resumed_runs: int,
    error_message: str | None = None,
) -> None:
    """Persist last discovery auto-halt reconciliation sweep metrics."""
    payload = {
        "started_at": started_at.astimezone(timezone.utc).isoformat(),
        "finished_at": finished_at.astimezone(timezone.utc).isoformat(),
        "status": status,
        "resumed_runs": max(0, int(resumed_runs)),
        "error_message": error_message,
    }
    ttl_seconds = max(300, int(settings.discovery_pipeline_halt_reconcile_interval_seconds) * 7)
    try:
        redis = get_redis_client()
        await redis.set(
            DISCOVERY_AUTO_HALT_RECONCILIATION_METRICS_KEY,
            json.dumps(payload),
            ex=ttl_seconds,
        )
    except Exception:
        logger.exception("Failed to write discovery auto-halt reconciliation metrics")


async def read_discovery_reconciliation_metrics() -> dict[str, Any] | None:
    """Load last discovery auto-halt reconciliation sweep metrics."""
    try:
        redis = get_redis_client()
        raw = await redis.get(DISCOVERY_AUTO_HALT_RECONCILIATION_METRICS_KEY)
    except Exception:
        logger.exception("Failed to read discovery auto-halt reconciliation metrics")
        return None
    if raw is None:
        return None
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        logger.warning(
            "Invalid discovery auto-halt reconciliation metrics payload",
            extra={"key": DISCOVERY_AUTO_HALT_RECONCILIATION_METRICS_KEY},
        )
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed
