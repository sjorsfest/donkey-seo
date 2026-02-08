"""Base class for pipeline step services."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar
import traceback

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.pipeline import StepExecution

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class StepResult(Generic[OutputT]):
    """Result of a step execution."""

    def __init__(
        self,
        success: bool,
        data: OutputT | None = None,
        error: str | None = None,
    ) -> None:
        self.success = success
        self.data = data
        self.error = error


class BaseStepService(ABC, Generic[InputT, OutputT]):
    """Abstract base class for all pipeline step services.

    Each step service should:
    1. Define step_number and step_name
    2. Implement _validate_preconditions to check prerequisites
    3. Implement _execute with the main step logic
    4. Implement _persist_results to save outputs
    """

    step_number: int
    step_name: str
    is_optional: bool = False
    requires_oauth: bool = False

    def __init__(
        self,
        session: AsyncSession,
        project_id: str,
        execution: StepExecution,
    ) -> None:
        self.session = session
        self.project_id = project_id
        self.execution = execution
        self._checkpoint: dict[str, Any] | None = None
        self._should_stop: bool = False

    async def run(self, input_data: InputT) -> StepResult[OutputT]:
        """Main execution method with error handling and progress tracking."""
        try:
            await self._update_status("running")

            # Check for checkpoint to resume
            if self.execution.checkpoint_data:
                await self._restore_checkpoint(self.execution.checkpoint_data)

            # Validate preconditions
            await self._validate_preconditions(input_data)

            # Execute main logic
            result = await self._execute(input_data)

            # Persist results
            await self._persist_results(result)

            await self._update_status("completed")
            return StepResult(success=True, data=result)

        except Exception as e:
            await self._handle_error(e)
            return StepResult(success=False, error=str(e))

    @abstractmethod
    async def _execute(self, input_data: InputT) -> OutputT:
        """Override with step-specific logic."""
        pass

    @abstractmethod
    async def _validate_preconditions(self, input_data: InputT) -> None:
        """Validate that prerequisites are met.

        Raises:
            StepPreconditionError: If preconditions are not met.
        """
        pass

    @abstractmethod
    async def _persist_results(self, result: OutputT) -> None:
        """Save results to database."""
        pass

    async def _update_progress(
        self,
        percent: float,
        message: str,
        items_processed: int = 0,
        items_total: int | None = None,
    ) -> None:
        """Update progress tracking."""
        self.execution.progress_percent = min(percent, 100.0)
        self.execution.progress_message = message
        self.execution.items_processed = items_processed
        if items_total is not None:
            self.execution.items_total = items_total
        await self.session.commit()

    async def _save_checkpoint(self, checkpoint_data: dict[str, Any]) -> None:
        """Save checkpoint for resumability."""
        self.execution.checkpoint_data = checkpoint_data
        await self.session.commit()

    async def _restore_checkpoint(self, checkpoint_data: dict[str, Any]) -> None:
        """Restore from checkpoint."""
        self._checkpoint = checkpoint_data

    async def _update_status(self, status: str) -> None:
        """Update execution status."""
        self.execution.status = status
        now = datetime.now(timezone.utc)
        if status == "running":
            self.execution.started_at = now
        elif status in ("completed", "failed", "skipped"):
            self.execution.completed_at = now
            if status == "completed":
                self.execution.progress_percent = 100.0
        await self.session.commit()

    async def _handle_error(self, error: Exception) -> None:
        """Handle and log errors."""
        self.execution.status = "failed"
        self.execution.completed_at = datetime.now(timezone.utc)
        self.execution.error_message = str(error)
        self.execution.error_traceback = traceback.format_exc()
        await self.session.commit()

    async def request_stop(self) -> None:
        """Request the step to stop gracefully."""
        self._should_stop = True

    async def check_should_stop(self) -> bool:
        """Check if the step should stop (for pause/cancel)."""
        return self._should_stop

    def set_result_summary(self, summary: dict[str, Any]) -> None:
        """Set the result summary for the execution."""
        self.execution.result_summary = summary
