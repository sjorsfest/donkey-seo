"""Base class for pipeline step services."""

import copy
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar
import traceback

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.brand import BrandProfile
from app.models.discovery_learning import DiscoveryIterationLearning
from app.models.generated_dtos import PipelineRunPatchDTO
from app.models.pipeline import PipelineRun, StepExecution
from app.models.project import Project
from app.services.discovery_capabilities import (
    capabilities_from_legacy_steps,
    normalize_capability_key,
)
from app.services.run_strategy import RunStrategy, resolve_run_strategy
from app.services.task_manager import TaskManager

logger = logging.getLogger(__name__)

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
    capability_key: str = "unknown"
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
        self._run_strategy: RunStrategy | None = None

    async def run(self, input_data: InputT) -> StepResult[OutputT]:
        """Main execution method with error handling and progress tracking."""
        step_info = {"step": self.step_number, "step_name": self.step_name, "project_id": self.project_id}
        logger.info("Step started", extra=step_info)

        try:
            await self._update_status("running")

            # Check for checkpoint to resume
            if self.execution.checkpoint_data:
                logger.info("Restoring from checkpoint", extra=step_info)
                await self._restore_checkpoint(self.execution.checkpoint_data)

            # Validate preconditions
            await self._validate_preconditions(input_data)
            logger.info("Preconditions validated", extra=step_info)

            # Execute main logic
            result = await self._execute(input_data)
            logger.info("Step execution finished, validating output", extra=step_info)

            # Validate output is usable downstream
            await self._validate_output(result, input_data)
            logger.info("Output validated, persisting results", extra=step_info)

            # Persist results
            await self._persist_results(result)

            await self._update_status("completed")
            logger.info("Step completed successfully", extra=step_info)
            return StepResult(success=True, data=result)

        except Exception as e:
            logger.warning(
                "Step failed",
                extra={**step_info, "error": str(e)},
            )
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

    async def _validate_output(self, result: OutputT, input_data: InputT) -> None:
        """Validate step output before persisting.

        Override in step services when the next step requires minimum output
        guarantees (e.g., non-empty seeds, keywords, or topics).
        """
        return None

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
        await self._publish_task_progress(percent=percent, message=message)

    async def _publish_task_progress(self, *, percent: float, message: str) -> None:
        """Best-effort mirror of step progress to task status payload."""
        pipeline_run_id = self.execution.pipeline_run_id
        if not pipeline_run_id:
            return

        task_manager = TaskManager()
        task_id = str(pipeline_run_id)
        try:
            current_task = await task_manager.get_task_status(task_id)
            run_progress = self._estimate_run_progress_percent(
                task_status=current_task,
                step_percent=percent,
            )
            await task_manager.set_task_state(
                task_id=task_id,
                status="running",
                stage=message,
                current_step=self.step_number,
                current_step_name=self.step_name,
                progress_percent=run_progress,
                error_message=None,
            )
        except Exception:
            logger.warning(
                "Failed to mirror step progress to task status",
                extra={
                    "project_id": self.project_id,
                    "pipeline_run_id": task_id,
                    "step_number": self.step_number,
                    "step_name": self.step_name,
                },
            )

    @staticmethod
    def _estimate_run_progress_percent(
        *,
        task_status: dict[str, Any] | None,
        step_percent: float,
    ) -> float | None:
        """Estimate run-level progress from completed steps + current step percent."""
        if not isinstance(task_status, dict):
            return None

        try:
            completed_steps = int(task_status.get("completed_steps") or 0)
            total_steps = int(task_status.get("total_steps") or 0)
        except (TypeError, ValueError):
            return None

        if total_steps <= 0:
            return None

        clamped_step = min(max(step_percent, 0.0), 100.0) / 100.0
        progress = ((completed_steps + clamped_step) / total_steps) * 100
        return round(min(progress, 100.0), 2)

    async def _save_checkpoint(self, checkpoint_data: dict[str, Any]) -> None:
        """Save checkpoint for resumability."""
        logger.info(
            "Saving checkpoint",
            extra={"step": self.step_number, "step_name": self.step_name, "checkpoint_keys": list(checkpoint_data.keys())},
        )
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

    async def get_run_strategy(self) -> RunStrategy:
        """Resolve run-scoped strategy for the current pipeline execution."""
        if self._run_strategy is not None:
            return self._run_strategy

        run_strategy_payload: dict[str, Any] | None = None
        if self.execution.pipeline_run_id:
            run_result = await self.session.execute(
                select(PipelineRun).where(PipelineRun.id == self.execution.pipeline_run_id)
            )
            run = run_result.scalar_one_or_none()
            if run and run.steps_config:
                run_strategy_payload = run.steps_config.get("strategy")

        brand_result = await self.session.execute(
            select(BrandProfile).where(BrandProfile.project_id == self.project_id)
        )
        brand = brand_result.scalar_one_or_none()

        project_result = await self.session.execute(
            select(Project).where(Project.id == self.project_id)
        )
        project = project_result.scalar_one_or_none()

        self._run_strategy = resolve_run_strategy(
            strategy_payload=run_strategy_payload,
            brand=brand,
            primary_goal=project.primary_goal if project else None,
        )
        return self._run_strategy

    async def _load_pipeline_run(self) -> PipelineRun | None:
        """Load the pipeline run for this step execution."""
        pipeline_run_id = self.execution.pipeline_run_id
        if not pipeline_run_id:
            return None
        result = await self.session.execute(
            select(PipelineRun).where(PipelineRun.id == pipeline_run_id)
        )
        return result.scalar_one_or_none()

    async def get_steps_config(self) -> dict[str, Any]:
        """Return mutable copy of run steps_config."""
        run = await self._load_pipeline_run()
        if not run or not isinstance(run.steps_config, dict):
            return {}
        return copy.deepcopy(run.steps_config)

    async def update_steps_config(
        self,
        updates: dict[str, Any],
        *,
        deep_merge: bool = True,
    ) -> dict[str, Any]:
        """Patch run steps_config safely and commit it."""
        run = await self._load_pipeline_run()
        if run is None:
            return {}

        existing = run.steps_config if isinstance(run.steps_config, dict) else {}
        if deep_merge:
            merged = self._deep_merge_dict(existing, updates)
        else:
            merged = {**existing, **updates}

        run.patch(
            self.session,
            PipelineRunPatchDTO.from_partial({"steps_config": merged}),
        )
        await self.session.commit()
        return merged

    async def get_market_diagnosis(self) -> dict[str, Any] | None:
        """Read current market diagnosis from steps_config."""
        steps_config = await self.get_steps_config()
        diagnosis = steps_config.get("market_diagnosis")
        if isinstance(diagnosis, dict):
            return diagnosis
        return None

    async def set_market_diagnosis(self, diagnosis: dict[str, Any]) -> dict[str, Any]:
        """Persist market diagnosis under steps_config.market_diagnosis."""
        return await self.update_steps_config({"market_diagnosis": diagnosis})

    async def get_market_mode(self, default: str = "mixed") -> str:
        """Resolve effective market mode for the current step run."""
        diagnosis = await self.get_market_diagnosis()
        if diagnosis and isinstance(diagnosis.get("mode"), str):
            return str(diagnosis["mode"])
        strategy = await self.get_run_strategy()
        override = getattr(strategy, "market_mode_override", "auto")
        if override and override != "auto":
            return str(override)
        return default

    async def build_learning_context(
        self,
        capability_key: str,
        agent_name: str | None = None,
        *,
        max_items: int = 8,
        max_chars: int = 2200,
    ) -> str:
        """Build compact project-memory guidance for discovery agent prompts."""
        normalized_capability = normalize_capability_key(capability_key)
        if normalized_capability is None:
            return ""

        result = await self.session.execute(
            select(DiscoveryIterationLearning)
            .where(DiscoveryIterationLearning.project_id == self.project_id)
            .order_by(DiscoveryIterationLearning.created_at.desc())
            .limit(200)
        )
        rows = list(result.scalars())
        if not rows:
            return ""

        selected: list[tuple[float, DiscoveryIterationLearning]] = []
        seen_keys: set[str] = set()

        for idx, row in enumerate(rows):
            if row.learning_key in seen_keys:
                continue

            capabilities = {
                normalize_capability_key(item)
                for item in (row.applies_to_capabilities or [])
            }
            capabilities.discard(None)
            capabilities |= self._legacy_capabilities_from_row(row)
            if normalized_capability not in capabilities:
                continue

            agents = {
                str(item).strip()
                for item in (row.applies_to_agents or [])
                if str(item).strip()
            }
            if agent_name and agents and agent_name not in agents:
                continue

            recency_bonus = max(0.0, 1.0 - (idx / max(len(rows), 1)))
            score = self._learning_priority_score(row) + recency_bonus
            selected.append((score, row))
            seen_keys.add(row.learning_key)

        if not selected:
            return ""

        selected.sort(key=lambda item: item[0], reverse=True)
        top_rows = [item[1] for item in selected[:max_items]]
        lines = ["Discovery memory signals from previous iterations:"]
        for row in top_rows:
            recommendation = str(row.recommendation or "").strip()
            suffix = f" -> {recommendation}" if recommendation else ""
            line = (
                f"- [{row.status}/{row.polarity}] {row.title}: {row.detail}{suffix}"
            )
            projected = "\n".join([*lines, line])
            if len(projected) > max_chars and len(lines) > 1:
                break
            lines.append(line)
        return "\n".join(lines)

    def _deep_merge_dict(self, current: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge dict updates into current dict."""
        merged = copy.deepcopy(current)
        for key, value in updates.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = self._deep_merge_dict(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _legacy_capabilities_from_row(self, row: DiscoveryIterationLearning) -> set[str]:
        """Dual-read compatibility for legacy step-based applicability payloads."""
        evidence = row.evidence if isinstance(row.evidence, dict) else {}
        legacy_steps = evidence.get("applies_to_steps")
        if isinstance(legacy_steps, list):
            return capabilities_from_legacy_steps(legacy_steps)

        # Handle transitional rows that may still expose an applies_to_steps attribute.
        transitional_steps = getattr(row, "applies_to_steps", None)
        if isinstance(transitional_steps, list):
            return capabilities_from_legacy_steps(transitional_steps)
        return set()

    def _learning_priority_score(self, row: DiscoveryIterationLearning) -> float:
        status_weight = {
            "regressed": 3.0,
            "new": 2.0,
            "confirmed": 1.0,
        }.get(str(row.status or "").lower(), 0.5)
        novelty = float(row.novelty_score) if row.novelty_score is not None else 0.0
        confidence = float(row.confidence) if row.confidence is not None else 0.0
        return status_weight + novelty + confidence
