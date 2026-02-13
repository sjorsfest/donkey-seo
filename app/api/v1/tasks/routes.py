"""Task API endpoints."""

from fastapi import APIRouter, HTTPException, status

from app.api.v1.tasks.constants import TASK_NOT_FOUND_DETAIL
from app.dependencies import CurrentUser
from app.schemas.task import TaskStatusResponse
from app.services.task_manager import TaskManager

router = APIRouter()


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str, _current_user: CurrentUser) -> TaskStatusResponse:
    """Get task status from Redis."""
    task_manager = TaskManager()
    task_status = await task_manager.get_task_status(task_id)

    if task_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=TASK_NOT_FOUND_DETAIL,
        )

    return TaskStatusResponse.model_validate(task_status)
