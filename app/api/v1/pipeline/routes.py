"""Pipeline API endpoints."""

import logging
import uuid

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.api.v1.dependencies import get_user_project
from app.api.v1.pipeline.constants import (
    CONTENT_PIPELINE_ALREADY_RUNNING_DETAIL,
    DEFAULT_RUN_LIMIT,
    DISCOVERY_PIPELINE_ALREADY_RUNNING_DETAIL,
    MAX_RUN_LIMIT,
    MULTIPLE_RUNNING_PIPELINES_DETAIL,
    NO_PAUSED_PIPELINE_DETAIL,
    NO_RUNNING_PIPELINE_DETAIL,
    PIPELINE_QUEUE_FULL_DETAIL,
    PIPELINE_RUN_NOT_FOUND_DETAIL,
)
from app.api.v1.pipeline.openapi_docs import PIPELINE_START_OPENAPI_EXTRA
from app.dependencies import CurrentUser, DbSession
from app.models.discovery_snapshot import DiscoveryTopicSnapshot
from app.models.generated_dtos import PipelineRunCreateDTO
from app.models.pipeline import PipelineRun
from app.schemas.pipeline import (
    ContentPipelineConfig,
    DiscoveryTopicSnapshotResponse,
    PipelineProgressResponse,
    PipelineRunResponse,
    PipelineStartRequest,
    StepExecutionResponse,
)
from app.services.pipeline_orchestrator import PipelineOrchestrator
from app.services.pipeline_task_manager import (
    PipelineQueueFullError,
    get_content_pipeline_task_manager,
    get_discovery_pipeline_task_manager,
)
from app.services.task_manager import TaskManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/{project_id}/start",
    response_model=PipelineRunResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start pipeline run",
    description="Create a new module pipeline run and queue background execution.",
    openapi_extra=PIPELINE_START_OPENAPI_EXTRA,
)
async def start_pipeline(
    project_id: uuid.UUID,
    request: PipelineStartRequest,
    current_user: CurrentUser,
    session: DbSession,
) -> PipelineRun:
    """Start a new pipeline run for a project."""
    project = await get_user_project(project_id, current_user, session)
    project_id_str = str(project_id)
    pipeline_module = request.mode

    if pipeline_module == "discovery":
        start_step = request.start_step or 1
        end_step = request.end_step or 8
        skip_steps = request.skip_steps or []
    else:
        start_step = request.start_step or 1
        end_step = request.end_step or 3
        skip_steps = request.skip_steps or []

    running_result = await session.execute(
        select(PipelineRun).where(
            PipelineRun.project_id == project_id,
            PipelineRun.pipeline_module == pipeline_module,
            PipelineRun.status == "running",
        )
    )
    if running_result.scalar_one_or_none():
        detail = (
            DISCOVERY_PIPELINE_ALREADY_RUNNING_DETAIL
            if pipeline_module == "discovery"
            else CONTENT_PIPELINE_ALREADY_RUNNING_DETAIL
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

    strategy_payload = request.strategy.model_dump() if request.strategy else None
    discovery_payload = request.discovery.model_dump() if request.discovery else None
    content_config = request.content
    if content_config is None:
        content_config = ContentPipelineConfig()
    content_payload = content_config.model_dump()
    step_inputs_payload: dict[str, dict[str, object]] = {}
    if pipeline_module == "content":
        step_inputs_payload["1"] = {
            "max_briefs": content_payload.get("max_briefs", 20),
            "posts_per_week": content_payload.get("posts_per_week", 1),
            "preferred_weekdays": content_payload.get("preferred_weekdays", []),
            "min_lead_days": content_payload.get("min_lead_days", 7),
            "publication_start_date": content_payload.get("publication_start_date"),
            "use_llm_timing_hints": content_payload.get("use_llm_timing_hints", True),
            "llm_timing_flex_days": content_payload.get("llm_timing_flex_days", 14),
            "include_zero_data_topics": content_payload.get("include_zero_data_topics", True),
            "zero_data_topic_share": content_payload.get("zero_data_topic_share", 0.2),
            "zero_data_fit_score_min": content_payload.get("zero_data_fit_score_min", 0.65),
        }

    steps_config = {
        "pipeline_module": pipeline_module,
        "start": start_step,
        "end": end_step,
        "skip": skip_steps,
        "strategy": strategy_payload,
        "primary_goal": project.primary_goal,
        "discovery": discovery_payload,
        "content": content_payload,
        "iteration_index": 0,
        "selected_topic_ids": [],
        "step_inputs": step_inputs_payload,
    }

    pipeline_run = PipelineRun.create(
        session,
        PipelineRunCreateDTO(
            project_id=project_id_str,
            pipeline_module=pipeline_module,
            status="pending",
            start_step=start_step,
            end_step=end_step,
            skip_steps=skip_steps,
            steps_config=steps_config,
        ),
    )
    await session.flush()
    await session.refresh(pipeline_run, ["step_executions"])
    await session.commit()

    task_id = str(pipeline_run.id)
    total_steps = len([step for step in range(start_step, end_step + 1) if step not in skip_steps])
    task_manager = TaskManager()
    await task_manager.set_task_state(
        task_id=task_id,
        status="queued",
        stage="Queued pipeline execution",
        project_id=project_id_str,
        pipeline_module=pipeline_module,
        source_topic_id=pipeline_run.source_topic_id,
        current_step=start_step,
        current_step_name=None,
        completed_steps=0,
        total_steps=total_steps,
        progress_percent=0.0,
        error_message=None,
    )

    queue = (
        get_discovery_pipeline_task_manager()
        if pipeline_module == "discovery"
        else get_content_pipeline_task_manager()
    )
    try:
        await queue.enqueue_start(
            project_id=project_id_str,
            run_id=task_id,
        )
    except PipelineQueueFullError as exc:
        pipeline_run.status = "paused"
        pipeline_run.error_message = PIPELINE_QUEUE_FULL_DETAIL
        await session.flush()
        await session.commit()
        await task_manager.set_task_state(
            task_id=task_id,
            status="paused",
            stage="Pipeline queue is full",
            error_message=PIPELINE_QUEUE_FULL_DETAIL,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=PIPELINE_QUEUE_FULL_DETAIL,
        ) from exc

    return pipeline_run


@router.get(
    "/{project_id}/runs",
    response_model=list[PipelineRunResponse],
    summary="List pipeline runs",
)
async def list_pipeline_runs(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
    limit: int = Query(DEFAULT_RUN_LIMIT, ge=1, le=MAX_RUN_LIMIT),
) -> list[PipelineRun]:
    await get_user_project(project_id, current_user, session)
    result = await session.execute(
        select(PipelineRun)
        .where(PipelineRun.project_id == project_id)
        .options(selectinload(PipelineRun.step_executions))
        .order_by(PipelineRun.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


@router.get(
    "/{project_id}/runs/{run_id}",
    response_model=PipelineRunResponse,
    summary="Get pipeline run",
)
async def get_pipeline_run(
    project_id: uuid.UUID,
    run_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> PipelineRun:
    await get_user_project(project_id, current_user, session)
    result = await session.execute(
        select(PipelineRun)
        .where(PipelineRun.id == run_id, PipelineRun.project_id == project_id)
        .options(selectinload(PipelineRun.step_executions))
    )
    pipeline_run = result.scalar_one_or_none()
    if not pipeline_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=PIPELINE_RUN_NOT_FOUND_DETAIL,
        )
    return pipeline_run


@router.get(
    "/{project_id}/runs/{run_id}/progress",
    response_model=PipelineProgressResponse,
    summary="Get pipeline progress",
)
async def get_pipeline_progress(
    project_id: uuid.UUID,
    run_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> PipelineProgressResponse:
    await get_user_project(project_id, current_user, session)
    result = await session.execute(
        select(PipelineRun)
        .where(PipelineRun.id == run_id, PipelineRun.project_id == project_id)
        .options(selectinload(PipelineRun.step_executions))
    )
    pipeline_run = result.scalar_one_or_none()
    if not pipeline_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=PIPELINE_RUN_NOT_FOUND_DETAIL,
        )

    steps = pipeline_run.step_executions
    if steps:
        completed = sum(1 for step in steps if step.status == "completed")
        total = len(steps)
        overall_progress = (completed / total) * 100 if total > 0 else 0
        current_step = next((step for step in steps if step.status == "running"), None)
    else:
        overall_progress = 0
        current_step = None

    return PipelineProgressResponse(
        run_id=str(pipeline_run.id),
        status=pipeline_run.status,
        current_step=current_step.step_number if current_step else None,
        current_step_name=current_step.step_name if current_step else None,
        overall_progress=overall_progress,
        steps=[StepExecutionResponse.model_validate(step) for step in steps],
    )


@router.get(
    "/{project_id}/runs/{run_id}/discovery-snapshots",
    response_model=list[DiscoveryTopicSnapshotResponse],
    summary="List discovery snapshots",
)
async def list_discovery_snapshots(
    project_id: uuid.UUID,
    run_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> list[DiscoveryTopicSnapshot]:
    await get_user_project(project_id, current_user, session)
    run_result = await session.execute(
        select(PipelineRun).where(
            PipelineRun.id == run_id,
            PipelineRun.project_id == project_id,
        )
    )
    pipeline_run = run_result.scalar_one_or_none()
    if not pipeline_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=PIPELINE_RUN_NOT_FOUND_DETAIL,
        )

    result = await session.execute(
        select(DiscoveryTopicSnapshot)
        .where(
            DiscoveryTopicSnapshot.project_id == project_id,
            DiscoveryTopicSnapshot.pipeline_run_id == run_id,
        )
        .order_by(
            DiscoveryTopicSnapshot.iteration_index.asc(),
            DiscoveryTopicSnapshot.created_at.asc(),
        )
    )
    return list(result.scalars().all())


@router.post(
    "/{project_id}/pause",
    status_code=status.HTTP_200_OK,
    summary="Pause running pipeline",
)
async def pause_pipeline(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> dict[str, str]:
    await get_user_project(project_id, current_user, session)
    result = await session.execute(
        select(PipelineRun).where(
            PipelineRun.project_id == project_id,
            PipelineRun.status == "running",
        )
    )
    runs = list(result.scalars().all())
    if not runs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=NO_RUNNING_PIPELINE_DETAIL,
        )
    if len(runs) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=MULTIPLE_RUNNING_PIPELINES_DETAIL,
        )

    pipeline_run = runs[0]
    orchestrator = PipelineOrchestrator(session, str(project_id))
    await orchestrator.pause_pipeline(str(pipeline_run.id))
    return {"message": "Pipeline paused"}


@router.post(
    "/{project_id}/runs/{run_id}/pause",
    status_code=status.HTTP_200_OK,
    summary="Pause run by id",
)
async def pause_pipeline_run(
    project_id: uuid.UUID,
    run_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> dict[str, str]:
    await get_user_project(project_id, current_user, session)
    orchestrator = PipelineOrchestrator(session, str(project_id))
    try:
        await orchestrator.pause_pipeline(str(run_id))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=NO_RUNNING_PIPELINE_DETAIL,
        ) from exc
    return {"message": "Pipeline paused"}


@router.post(
    "/{project_id}/resume/{run_id}",
    status_code=status.HTTP_200_OK,
    summary="Resume paused pipeline",
)
async def resume_pipeline(
    project_id: uuid.UUID,
    run_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> dict[str, str]:
    await get_user_project(project_id, current_user, session)
    result = await session.execute(
        select(PipelineRun).where(
            PipelineRun.id == run_id,
            PipelineRun.project_id == project_id,
            PipelineRun.status == "paused",
        )
    )
    pipeline_run = result.scalar_one_or_none()
    if not pipeline_run:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=NO_PAUSED_PIPELINE_DETAIL,
        )

    task_manager = TaskManager()
    await task_manager.set_task_state(
        task_id=str(pipeline_run.id),
        status="queued",
        stage="Queued pipeline resume",
        project_id=str(project_id),
        pipeline_module=pipeline_run.pipeline_module,
        source_topic_id=pipeline_run.source_topic_id,
        current_step=pipeline_run.paused_at_step,
        current_step_name=None,
        error_message=None,
    )

    queue = (
        get_discovery_pipeline_task_manager()
        if pipeline_run.pipeline_module == "discovery"
        else get_content_pipeline_task_manager()
    )
    try:
        await queue.enqueue_resume(
            project_id=str(project_id),
            run_id=str(pipeline_run.id),
        )
    except PipelineQueueFullError as exc:
        await task_manager.set_task_state(
            task_id=str(pipeline_run.id),
            status="paused",
            stage="Pipeline queue is full",
            error_message=PIPELINE_QUEUE_FULL_DETAIL,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=PIPELINE_QUEUE_FULL_DETAIL,
        ) from exc

    return {"message": "Pipeline resumed"}
