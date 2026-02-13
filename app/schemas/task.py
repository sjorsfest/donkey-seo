"""Task schemas."""

from datetime import datetime

from pydantic import BaseModel


class TaskStatusResponse(BaseModel):
    """Schema for task status response."""

    task_id: str
    status: str
    stage: str | None = None
    project_id: str | None = None
    current_step: int | None = None
    current_step_name: str | None = None
    completed_steps: int = 0
    total_steps: int | None = None
    progress_percent: float | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
