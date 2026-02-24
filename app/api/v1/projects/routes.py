"""Project API endpoints."""

import logging
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select

from app.api.v1.dependencies import get_user_project
from app.api.v1.pipeline.constants import PIPELINE_QUEUE_FULL_DETAIL
from app.api.v1.projects.constants import DEFAULT_PAGE, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from app.dependencies import CurrentUser, DbSession
from app.models.generated_dtos import PipelineRunCreateDTO, ProjectCreateDTO, ProjectPatchDTO
from app.models.pipeline import PipelineRun
from app.models.project import Project
from app.schemas.project import (
    ProjectCreate,
    ProjectListResponse,
    ProjectOnboardingBootstrapRequest,
    ProjectOnboardingBootstrapResponse,
    ProjectResponse,
    ProjectUpdate,
)
from app.schemas.task import TaskStatusResponse
from app.services.pipeline_task_manager import (
    PipelineQueueFullError,
    get_setup_pipeline_task_manager,
)
from app.services.publication_webhook import (
    backfill_project_publication_webhook_deliveries,
)
from app.services.task_manager import TaskManager

logger = logging.getLogger(__name__)

router = APIRouter()


@dataclass(frozen=True)
class _ProjectSettingsCreateFields:
    skip_steps: list[int] | None
    notification_webhook: str | None
    notification_webhook_secret: str | None


def _project_settings_create_fields(
    payload: ProjectCreate | ProjectOnboardingBootstrapRequest,
) -> _ProjectSettingsCreateFields:
    """Flatten project settings into DTO kwargs for creation."""
    settings_payload = payload.settings
    if settings_payload is None:
        return _ProjectSettingsCreateFields(
            skip_steps=None,
            notification_webhook=None,
            notification_webhook_secret=None,
        )
    return _ProjectSettingsCreateFields(
        skip_steps=settings_payload.skip_steps,
        notification_webhook=settings_payload.notification_webhook,
        notification_webhook_secret=settings_payload.notification_webhook_secret,
    )


def _project_settings_patch_fields(settings_payload: dict[str, object] | None) -> tuple[dict[str, object], bool]:
    """Extract patchable project settings fields and webhook-touch marker."""
    if not isinstance(settings_payload, dict):
        return {}, False

    patch_fields: dict[str, object] = {}
    webhook_config_touched = False
    if "skip_steps" in settings_payload:
        patch_fields["skip_steps"] = settings_payload.get("skip_steps")
    if "notification_webhook" in settings_payload:
        patch_fields["notification_webhook"] = settings_payload.get("notification_webhook")
        webhook_config_touched = True
    if "notification_webhook_secret" in settings_payload:
        patch_fields["notification_webhook_secret"] = settings_payload.get(
            "notification_webhook_secret"
        )
        webhook_config_touched = True
    return patch_fields, webhook_config_touched


@router.post(
    "/",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create project",
    description=(
        "Create a keyword research project with domain, locale, goals, and optional pipeline "
        "settings."
    ),
)
async def create_project(
    project_data: ProjectCreate,
    current_user: CurrentUser,
    session: DbSession,
) -> Project:
    """Create a new keyword research project."""
    project = await _create_project(
        session=session,
        user_id=current_user.id,
        payload=project_data,
    )

    logger.info(
        "Project created",
        extra={
            "project_id": str(project.id),
            "domain": project.domain,
            "user_id": str(current_user.id),
        },
    )

    return project


@router.post(
    "/onboarding/bootstrap",
    response_model=ProjectOnboardingBootstrapResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Bootstrap onboarding project",
    description=(
        "Create a project and immediately queue setup pipeline steps 1-5 "
        "for domain + brand extraction."
    ),
)
async def bootstrap_onboarding_project(
    request: ProjectOnboardingBootstrapRequest,
    current_user: CurrentUser,
    session: DbSession,
) -> ProjectOnboardingBootstrapResponse:
    """Create a project and kick off setup pipeline asynchronously."""
    project = await _create_project(
        session=session,
        user_id=current_user.id,
        payload=request,
    )

    strategy_payload = request.strategy.model_dump() if request.strategy else None
    steps_config = {
        "pipeline_module": "setup",
        "start": 1,
        "end": 5,
        "skip": [],
        "strategy": strategy_payload,
        "primary_goal": project.primary_goal,
        "discovery": None,
        "content": None,
        "iteration_index": 0,
        "selected_topic_ids": [],
        "step_inputs": {},
    }

    setup_run = PipelineRun.create(
        session,
        PipelineRunCreateDTO(
            project_id=str(project.id),
            pipeline_module="setup",
            status="pending",
            start_step=1,
            end_step=5,
            skip_steps=[],
            steps_config=steps_config,
        ),
    )
    await session.commit()

    task_manager = TaskManager()
    task_payload = await task_manager.set_task_state(
        task_id=str(setup_run.id),
        status="queued",
        stage="Queued setup bootstrap",
        project_id=str(project.id),
        pipeline_module="setup",
        source_topic_id=setup_run.source_topic_id,
        current_step=1,
        current_step_name="setup_project",
        completed_steps=0,
        total_steps=5,
        progress_percent=0.0,
        error_message=None,
    )

    try:
        await get_setup_pipeline_task_manager().enqueue_start(
            project_id=str(project.id),
            run_id=str(setup_run.id),
        )
    except PipelineQueueFullError as exc:
        setup_run.status = "paused"
        setup_run.error_message = PIPELINE_QUEUE_FULL_DETAIL
        await session.flush()
        await session.commit()
        await task_manager.set_task_state(
            task_id=str(setup_run.id),
            status="paused",
            stage="Pipeline queue is full",
            error_message=PIPELINE_QUEUE_FULL_DETAIL,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=PIPELINE_QUEUE_FULL_DETAIL,
        ) from exc

    return ProjectOnboardingBootstrapResponse(
        project=ProjectResponse.model_validate(project),
        setup_run_id=str(setup_run.id),
        setup_task=TaskStatusResponse.model_validate(task_payload),
    )


@router.get(
    "/",
    response_model=ProjectListResponse,
    summary="List projects",
    description="Return paginated projects for the authenticated user, ordered by most recent.",
)
async def list_projects(
    current_user: CurrentUser,
    session: DbSession,
    page: int = Query(DEFAULT_PAGE, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
) -> ProjectListResponse:
    """List all projects for the current user."""
    count_query = (
        select(func.count())
        .select_from(Project)
        .where(Project.user_id == current_user.id)
    )
    total = await session.scalar(count_query) or 0

    offset = (page - 1) * page_size
    query = (
        select(Project)
        .where(Project.user_id == current_user.id)
        .order_by(Project.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    result = await session.execute(query)
    projects = result.scalars().all()

    return ProjectListResponse(
        items=[ProjectResponse.model_validate(p) for p in projects],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{project_id}",
    response_model=ProjectResponse,
    summary="Get project",
    description="Return a single project by ID when it belongs to the authenticated user.",
)
async def get_project(
    project_id: str,
    current_user: CurrentUser,
    session: DbSession,
) -> Project:
    """Get a specific project by ID."""
    return await get_user_project(project_id, current_user, session)


@router.put(
    "/{project_id}",
    response_model=ProjectResponse,
    summary="Update project",
    description="Apply partial updates to editable fields on an existing project.",
)
async def update_project(
    project_id: str,
    project_data: ProjectUpdate,
    current_user: CurrentUser,
    session: DbSession,
) -> Project:
    """Update a project."""
    project = await get_user_project(project_id, current_user, session)

    update_data = project_data.model_dump(exclude_unset=True)
    settings_payload = update_data.get("settings")
    patch_data = {
        field: value
        for field, value in update_data.items()
        if hasattr(project, field)
    }
    settings_patch_data, webhook_config_touched = _project_settings_patch_fields(
        settings_payload if isinstance(settings_payload, dict) else None
    )
    patch_data.update(settings_patch_data)
    project.patch(
        session,
        ProjectPatchDTO.from_partial(patch_data),
    )

    if webhook_config_touched:
        await backfill_project_publication_webhook_deliveries(
            session,
            project_id=str(project.id),
        )

    await session.flush()
    await session.refresh(project)

    return project


@router.delete(
    "/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete project",
    description="Delete a project owned by the authenticated user.",
)
async def delete_project(
    project_id: str,
    current_user: CurrentUser,
    session: DbSession,
) -> None:
    """Delete a project."""
    project = await get_user_project(project_id, current_user, session)

    logger.info(
        "Project deleted",
        extra={"project_id": str(project_id), "user_id": str(current_user.id)},
    )

    await project.delete(session)


async def _create_project(
    *,
    session: DbSession,
    user_id: str,
    payload: ProjectCreate | ProjectOnboardingBootstrapRequest,
) -> Project:
    """Persist a project from either standard create or onboarding bootstrap payload."""
    settings_fields = _project_settings_create_fields(payload)
    project = Project.create(
        session,
        ProjectCreateDTO(
            user_id=user_id,
            name=payload.name,
            domain=payload.domain,
            description=payload.description,
            primary_language=payload.primary_language,
            primary_locale=payload.primary_locale,
            secondary_locales=payload.secondary_locales,
            primary_goal=payload.goals.primary_objective if payload.goals else None,
            secondary_goals=payload.goals.secondary_goals if payload.goals else None,
            skip_steps=settings_fields.skip_steps,
            notification_webhook=settings_fields.notification_webhook,
            notification_webhook_secret=settings_fields.notification_webhook_secret,
        ),
    )

    if settings_fields.notification_webhook or settings_fields.notification_webhook_secret:
        await backfill_project_publication_webhook_deliveries(
            session,
            project_id=str(project.id),
        )

    await session.flush()
    await session.refresh(project)
    return project
