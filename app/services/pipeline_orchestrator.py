"""Pipeline orchestrator for coordinating step execution."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session_context
from app.core.exceptions import (
    PipelineAlreadyRunningError,
    StepNotFoundError,
    StepPreconditionError,
)
from app.models.generated_dtos import (
    PipelineRunCreateDTO,
    PipelineRunPatchDTO,
    StepExecutionCreateDTO,
)
from app.models.pipeline import PipelineRun, StepExecution
from app.models.project import Project
from app.schemas.pipeline import ContentPipelineConfig, PipelineMode
from app.services.discovery_loop import DiscoveryLoopResult, DiscoveryLoopSupervisor
from app.services.steps.base_step import BaseStepService
from app.services.steps.step_00_setup import SetupInput, Step00SetupService
from app.services.steps.step_01_brand import BrandInput, Step01BrandService
from app.services.steps.step_02_seeds import SeedsInput, Step02SeedsService
from app.services.steps.step_03_expansion import ExpansionInput, Step03ExpansionService
from app.services.steps.step_04_metrics import MetricsInput, Step04MetricsService
from app.services.steps.step_05_intent import IntentInput, Step05IntentService
from app.services.steps.step_06_clustering import ClusteringInput, Step06ClusteringService
from app.services.steps.step_07_prioritization import (
    PrioritizationInput,
    Step07PrioritizationService,
)
from app.services.steps.step_08_serp import (
    SerpValidationInput,
    Step08SerpValidationService,
)
from app.services.steps.step_12_brief import BriefInput, Step12BriefService
from app.services.steps.step_13_templates import Step13TemplatesService, TemplatesInput
from app.services.steps.step_14_article_writer import (
    ArticleWriterInput,
    Step14ArticleWriterService,
)
from app.services.task_manager import TaskManager

logger = logging.getLogger(__name__)


# Step names mapping
STEP_NAMES = {
    0: "setup",
    1: "brand_profile",
    2: "seed_topics",
    3: "keyword_expansion",
    4: "keyword_metrics",
    5: "intent_labeling",
    6: "clustering",
    7: "prioritization",
    8: "serp_validation",
    9: "content_inventory",
    10: "cannibalization",
    11: "internal_linking",
    12: "content_brief",
    13: "writer_templates",
    14: "article_generation",
}

# Step dependencies
STEP_DEPENDENCIES = {
    0: [],
    1: [0],
    2: [1],
    3: [2],
    4: [3],
    5: [4],
    6: [5],
    7: [6],
    8: [7],
    9: [0],
    10: [6, 9],
    11: [10],
    12: [7],
    13: [12],
    14: [13],
}

# Optional steps (can be skipped)
OPTIONAL_STEPS = {8, 9, 10, 11}
CONTENT_PRODUCTION_STEPS = (12, 13, 14)


class PipelineOrchestrator:
    """Orchestrates the execution of pipeline steps."""

    def __init__(self, session: AsyncSession, project_id: str) -> None:
        self.session = session
        self.project_id = project_id
        self.task_manager = TaskManager()
        self._current_run: PipelineRun | None = None

    async def start_pipeline(
        self,
        start_step: int = 0,
        end_step: int | None = None,
        skip_steps: list[int] | None = None,
        run_id: str | None = None,
        steps_config: dict[str, Any] | None = None,
    ) -> PipelineRun:
        """Start a new pipeline run."""
        end_step = end_step or 14
        skip_steps = skip_steps or []
        run_uuid = uuid.UUID(run_id) if run_id else None

        logger.info(
            "Starting pipeline",
            extra={
                "project_id": self.project_id,
                "run_id": run_id,
                "start_step": start_step,
                "end_step": end_step,
                "skip_steps": skip_steps,
                "mode": (steps_config or {}).get("mode"),
            },
        )

        # Check for running pipeline
        running_query = select(PipelineRun).where(
            PipelineRun.project_id == self.project_id,
            PipelineRun.status == "running",
        )
        if run_uuid is not None:
            running_query = running_query.where(PipelineRun.id != run_uuid)

        result = await self.session.execute(running_query)
        if result.scalar_one_or_none():
            raise PipelineAlreadyRunningError(self.project_id)

        if run_uuid is None:
            effective_steps_config = self._build_effective_steps_config(
                start_step=start_step,
                end_step=end_step,
                skip_steps=skip_steps,
                existing_config=steps_config,
            )
            run = PipelineRun.create(
                self.session,
                PipelineRunCreateDTO(
                    project_id=self.project_id,
                    status="running",
                    started_at=datetime.now(timezone.utc),
                    start_step=start_step,
                    end_step=end_step,
                    skip_steps=skip_steps,
                    steps_config=effective_steps_config,
                ),
            )
        else:
            result = await self.session.execute(
                select(PipelineRun).where(
                    PipelineRun.id == run_uuid,
                    PipelineRun.project_id == self.project_id,
                )
            )
            run = result.scalar_one_or_none()
            if run is None:
                raise ValueError(f"Pipeline run not found: {run_id}")
            effective_steps_config = self._build_effective_steps_config(
                start_step=start_step,
                end_step=end_step,
                skip_steps=skip_steps,
                existing_config=run.steps_config,
            )
            if steps_config:
                effective_steps_config = self._build_effective_steps_config(
                    start_step=start_step,
                    end_step=end_step,
                    skip_steps=skip_steps,
                    existing_config={**effective_steps_config, **steps_config},
                )

            run.patch(
                self.session,
                PipelineRunPatchDTO.from_partial({
                    "status": "running",
                    "started_at": datetime.now(timezone.utc),
                    "completed_at": None,
                    "error_message": None,
                    "paused_at_step": None,
                    "start_step": start_step,
                    "end_step": end_step,
                    "skip_steps": skip_steps,
                    "steps_config": effective_steps_config,
                }),
            )

        # Commit so per-step sessions can see the FK
        await self.session.commit()
        self._current_run = run
        mode = self._resolve_mode(run.steps_config)
        run_start, run_end = self._resolve_step_window(
            mode=mode,
            requested_start=start_step,
            requested_end=end_step,
        )

        task_id = str(run.id)
        total_steps = len(
            [step for step in range(run_start, run_end + 1) if step not in skip_steps]
        )
        await self.task_manager.set_task_state(
            task_id=task_id,
            status="running",
            stage="Pipeline started",
            project_id=self.project_id,
            current_step=run_start,
            current_step_name=STEP_NAMES.get(run_start),
            completed_steps=0,
            total_steps=total_steps,
            progress_percent=0.0,
            error_message=None,
        )

        try:
            if mode == "discovery_loop":
                result = await self._run_discovery_loop(run)
                await self._update_run_status(
                    run.id,
                    status="completed",
                    completed_at=datetime.now(timezone.utc),
                    paused_at_step=None,
                    error_message=None,
                )
                run.status = "completed"
                run.completed_at = datetime.now(timezone.utc)
                await self.task_manager.set_task_state(
                    task_id=task_id,
                    status="completed",
                    stage=(
                        "Discovery loop completed "
                        f"({result.accepted_count}/{result.target_count} accepted topics)"
                    ),
                    current_step=8,
                    current_step_name=STEP_NAMES.get(8),
                    progress_percent=100.0,
                    error_message=None,
                )
                if result.success and self._discovery_auto_start_enabled(run.steps_config):
                    await self._start_content_pipeline_from_discovery(
                        accepted_topic_ids=result.accepted_topic_ids,
                        content_config=result.content_config,
                        base_steps_config=run.steps_config or {},
                    )
                return run

            await self._run_step_range(
                run=run,
                start_step=run_start,
                end_step=run_end,
                skip_steps=skip_steps,
                task_id=task_id,
            )

            await self._update_run_status(
                run.id, status="completed", completed_at=datetime.now(timezone.utc)
            )
            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc)
            logger.info(
                "Pipeline completed", extra={"project_id": self.project_id, "run_id": str(run.id)}
            )

            await self.task_manager.set_task_state(
                task_id=task_id,
                status="completed",
                stage="Pipeline completed",
                current_step=run_end,
                current_step_name=STEP_NAMES.get(run_end),
                completed_steps=total_steps,
                progress_percent=100.0,
                error_message=None,
            )

        except Exception as e:
            step_num = run_start
            if isinstance(e, RuntimeError) and run.paused_at_step:
                step_num = run.paused_at_step
            try:
                await self._update_run_status(
                    run.id, status="paused", error_message=str(e), paused_at_step=step_num
                )
            except Exception:
                logger.error("Failed to persist pipeline error status to DB")
            run.status = "paused"
            run.error_message = str(e)
            run.paused_at_step = step_num
            logger.warning(
                "Pipeline paused due to error",
                extra={"project_id": self.project_id, "step": step_num, "error": str(e)},
            )

            await self.task_manager.set_task_state(
                task_id=task_id,
                status="paused",
                stage=(
                    "Pipeline paused at step "
                    f"{step_num}: {STEP_NAMES.get(step_num, f'step_{step_num}')}"
                ),
                current_step=step_num,
                current_step_name=STEP_NAMES.get(step_num),
                error_message=str(e),
            )

        return run

    async def run_single_step(self, step_number: int) -> StepExecution:
        """Run a single step independently."""
        if step_number not in STEP_NAMES:
            raise StepNotFoundError(step_number)

        logger.info(
            "Running single step",
            extra={
                "project_id": self.project_id,
                "step": step_number,
                "step_name": STEP_NAMES.get(step_number),
            },
        )

        # Verify dependencies
        await self._verify_dependencies(step_number)

        # Get or create pipeline run
        run = await self._get_or_create_run()
        task_id = str(run.id)
        await self.task_manager.set_task_state(
            task_id=task_id,
            status="running",
            stage=f"Running step {step_number}: {STEP_NAMES[step_number]}",
            project_id=self.project_id,
            current_step=step_number,
            current_step_name=STEP_NAMES.get(step_number),
            completed_steps=0,
            total_steps=1,
            progress_percent=0.0,
            error_message=None,
        )

        execution = await self._execute_step(run, step_number)
        if execution.status == "completed":
            await self.task_manager.set_task_state(
                task_id=task_id,
                status="completed",
                stage=f"Completed step {step_number}: {STEP_NAMES[step_number]}",
                current_step=step_number,
                current_step_name=STEP_NAMES.get(step_number),
                completed_steps=1,
                total_steps=1,
                progress_percent=100.0,
                error_message=None,
            )
        return execution

    async def resume_pipeline(self, run_id: str) -> PipelineRun:
        """Resume a paused pipeline."""
        logger.info("Resuming pipeline", extra={"project_id": self.project_id, "run_id": run_id})
        result = await self.session.execute(
            select(PipelineRun).where(
                PipelineRun.id == run_id,
                PipelineRun.project_id == self.project_id,
                PipelineRun.status == "paused",
            )
        )
        run = result.scalar_one_or_none()

        if not run:
            raise ValueError("No paused pipeline found")

        mode = self._resolve_mode(run.steps_config)
        if mode == "discovery_loop":
            return await self.start_pipeline(
                start_step=run.start_step or 2,
                end_step=run.end_step or 8,
                skip_steps=run.skip_steps or [],
                run_id=run_id,
                steps_config=run.steps_config,
            )

        run.status = "running"
        self._current_run = run

        # Find last completed step and resume from next
        last_completed = await self._get_last_completed_step(run)
        start_step = last_completed + 1 if last_completed is not None else 0
        end_step = run.end_step or 14
        skip_steps = run.skip_steps or []
        logger.info(
            "Last completed step: %s",
            last_completed,
            extra={
                "project_id": self.project_id,
                "run_id": run_id,
                "start_step": start_step,
                "end_step": end_step,
                "skip_steps": skip_steps,
            },
        )

        # Clear previous error when resuming
        run.error_message = None
        run.paused_at_step = None

        await self.session.commit()  # Commit status change before steps run

        task_id = str(run.id)
        existing_status = await self.task_manager.get_task_status(task_id)
        total_steps = len(
            [step for step in range(start_step, end_step + 1) if step not in skip_steps]
        )
        completed_steps = (
            await self._count_completed_steps(run.id) if existing_status is None else None
        )
        progress_percent = None
        if completed_steps is not None and total_steps > 0:
            progress_percent = round((completed_steps / total_steps) * 100, 2)

        await self.task_manager.set_task_state(
            task_id=task_id,
            status="running",
            stage="Pipeline resumed",
            project_id=self.project_id,
            current_step=start_step,
            current_step_name=STEP_NAMES.get(start_step),
            completed_steps=completed_steps,
            total_steps=total_steps,
            progress_percent=progress_percent,
            error_message=None,
        )

        step_num = start_step
        try:
            for step_num in range(start_step, end_step + 1):
                if await self._should_skip_step(step_num, skip_steps):
                    await self._set_step_skipped_task_state(task_id, step_num)
                    continue
                logger.info(
                    "Executing step %s",
                    step_num,
                    extra={
                        "project_id": self.project_id,
                        "run_id": run_id,
                        "step": step_num,
                        "step_name": STEP_NAMES.get(step_num),
                    },
                )
                execution = await self._execute_step(run, step_num)

                if execution.status == "completed":
                    await self.task_manager.mark_step_completed(
                        task_id=task_id,
                        step_number=step_num,
                        step_name=STEP_NAMES.get(step_num, f"step_{step_num}"),
                    )
                elif execution.status == "skipped":
                    await self._set_step_skipped_task_state(task_id, step_num)

            await self._update_run_status(
                run.id, status="completed", completed_at=datetime.now(timezone.utc)
            )
            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc)

            await self.task_manager.set_task_state(
                task_id=task_id,
                status="completed",
                stage="Pipeline completed",
                current_step=end_step,
                current_step_name=STEP_NAMES.get(end_step),
                progress_percent=100.0,
                error_message=None,
            )

        except Exception as e:
            try:
                await self._update_run_status(
                    run.id, status="paused", error_message=str(e), paused_at_step=step_num
                )
            except Exception:
                logger.error("Failed to persist pipeline error status to DB")
            run.status = "paused"
            run.error_message = str(e)
            run.paused_at_step = step_num
            logger.warning(
                "Resumed pipeline paused due to error",
                extra={"project_id": self.project_id, "step": step_num, "error": str(e)},
            )

            await self.task_manager.set_task_state(
                task_id=task_id,
                status="paused",
                stage=(
                    "Pipeline paused at step "
                    f"{step_num}: {STEP_NAMES.get(step_num, f'step_{step_num}')}"
                ),
                current_step=step_num,
                current_step_name=STEP_NAMES.get(step_num),
                error_message=str(e),
            )

        return run

    async def pause_pipeline(self) -> None:
        """Pause the current pipeline."""
        if self._current_run:
            logger.info(
                "Pausing pipeline",
                extra={"project_id": self.project_id, "run_id": str(self._current_run.id)},
            )
            self._current_run.status = "paused"
            await self.session.commit()

            await self.task_manager.set_task_state(
                task_id=str(self._current_run.id),
                status="paused",
                stage="Pipeline paused",
            )

    async def _run_step_range(
        self,
        *,
        run: PipelineRun,
        start_step: int,
        end_step: int,
        skip_steps: list[int],
        task_id: str,
    ) -> None:
        """Execute a contiguous step range and update task state."""
        for step_num in range(start_step, end_step + 1):
            if await self._should_skip_step(step_num, skip_steps):
                logger.info(
                    "Skipping step",
                    extra={
                        "project_id": self.project_id,
                        "step": step_num,
                        "step_name": STEP_NAMES.get(step_num),
                    },
                )
                await self._create_skipped_execution(run, step_num)
                await self._set_step_skipped_task_state(task_id, step_num)
                continue

            logger.info(
                "Executing step",
                extra={
                    "project_id": self.project_id,
                    "step": step_num,
                    "step_name": STEP_NAMES.get(step_num),
                },
            )
            execution = await self._execute_step(run, step_num)
            run.paused_at_step = step_num

            if execution.status == "completed":
                await self.task_manager.mark_step_completed(
                    task_id=task_id,
                    step_number=step_num,
                    step_name=STEP_NAMES.get(step_num, f"step_{step_num}"),
                )
            elif execution.status == "skipped":
                await self._set_step_skipped_task_state(task_id, step_num)

    async def _run_discovery_loop(self, run: PipelineRun) -> DiscoveryLoopResult:
        """Execute adaptive discovery iterations until enough topics are accepted."""
        supervisor = DiscoveryLoopSupervisor(
            session=self.session,
            project_id=self.project_id,
            run=run,
            task_manager=self.task_manager,
        )
        return await supervisor.run_loop(
            execute_step=self._execute_step,
            mark_step_completed=self.task_manager.mark_step_completed,
        )

    async def _start_content_pipeline_from_discovery(
        self,
        *,
        accepted_topic_ids: list[str],
        content_config: ContentPipelineConfig,
        base_steps_config: dict[str, Any],
    ) -> None:
        """Auto-start content production with selected accepted topics."""
        step_inputs = {
            "12": {
                "topic_ids": accepted_topic_ids,
                "max_briefs": content_config.max_briefs,
                "posts_per_week": content_config.posts_per_week,
                "preferred_weekdays": content_config.preferred_weekdays,
                "min_lead_days": content_config.min_lead_days,
                "publication_start_date": content_config.publication_start_date,
                "use_llm_timing_hints": content_config.use_llm_timing_hints,
                "llm_timing_flex_days": content_config.llm_timing_flex_days,
            }
        }
        chained_config = {
            "mode": "content_production",
            "strategy": base_steps_config.get("strategy"),
            "content": content_config.model_dump(),
            "discovery": base_steps_config.get("discovery"),
            "iteration_index": base_steps_config.get("iteration_index", 0),
            "selected_topic_ids": accepted_topic_ids,
            "step_inputs": step_inputs,
        }
        logger.info(
            "Auto-starting content production",
            extra={
                "project_id": self.project_id,
                "accepted_topics": len(accepted_topic_ids),
                "max_briefs": content_config.max_briefs,
            },
        )
        await self.start_pipeline(
            start_step=12,
            end_step=14,
            skip_steps=[],
            steps_config=chained_config,
        )

    def _build_effective_steps_config(
        self,
        *,
        start_step: int,
        end_step: int,
        skip_steps: list[int],
        existing_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        config = dict(existing_config or {})
        config["start"] = start_step
        config["end"] = end_step
        config["skip"] = skip_steps
        config.setdefault("mode", "full")
        config.setdefault("iteration_index", 0)
        config.setdefault("selected_topic_ids", [])
        config.setdefault("step_inputs", {})
        return config

    def _resolve_mode(self, steps_config: dict[str, Any] | None) -> PipelineMode:
        if not steps_config:
            return "full"
        mode = steps_config.get("mode", "full")
        if mode in {"full", "discovery_loop", "content_production"}:
            return mode
        return "full"

    def _resolve_step_window(
        self,
        *,
        mode: PipelineMode,
        requested_start: int,
        requested_end: int,
    ) -> tuple[int, int]:
        if mode == "discovery_loop":
            return 2, 8
        if mode == "content_production":
            return 12, 14
        return requested_start, requested_end

    def _discovery_auto_start_enabled(self, steps_config: dict[str, Any] | None) -> bool:
        if not steps_config:
            return True
        discovery_cfg = steps_config.get("discovery")
        if not isinstance(discovery_cfg, dict):
            return True
        return bool(discovery_cfg.get("auto_start_content", True))

    async def _should_skip_step(self, step_number: int, skip_steps: list[int]) -> bool:
        """Check if a step should be skipped."""
        if step_number in skip_steps:
            return True

        if step_number in OPTIONAL_STEPS:
            # Check skip logic for optional steps
            if step_number == 9:
                # Skip content inventory if no existing content
                return not await self._has_existing_content()
            if step_number == 10:
                # Skip cannibalization if step 9 was skipped
                return 9 in skip_steps
            if step_number == 11:
                # Skip linking if steps 9 and 10 were skipped
                return 9 in skip_steps and 10 in skip_steps
        return False

    async def _verify_dependencies(self, step_number: int) -> None:
        """Verify all dependencies are satisfied."""
        deps = STEP_DEPENDENCIES.get(step_number, [])
        for dep in deps:
            if not await self._is_step_completed(dep):
                raise StepPreconditionError(
                    step_number,
                    f"Step {dep} ({STEP_NAMES[dep]}) must be completed first",
                )

    async def _is_step_completed(self, step_number: int) -> bool:
        """Check if a step has been completed for this project."""
        result = await self.session.execute(
            select(StepExecution).where(
                StepExecution.step_number == step_number,
                StepExecution.status == "completed",
            )
        )
        return result.scalar_one_or_none() is not None

    async def _execute_step(self, run: PipelineRun, step_number: int) -> StepExecution:
        """Execute a single step with its own database session.

        Each step gets a fresh DB connection so long-running LLM calls
        in a previous step can't leave behind a stale connection.
        """
        async with get_session_context() as step_session:
            execution = StepExecution.create(
                step_session,
                StepExecutionCreateDTO(
                    pipeline_run_id=str(run.id),
                    step_number=step_number,
                    step_name=STEP_NAMES[step_number],
                    status="pending",
                ),
            )
            logger.info("Executing step %s: %s", step_number, STEP_NAMES[step_number])
            await step_session.flush()

            service = await self._get_step_service(step_number, execution, step_session)

            if service is None:
                logger.info("Step %s: %s not implemented", step_number, STEP_NAMES[step_number])
                execution.status = "skipped"
                execution.progress_message = "Step not implemented"
                return execution

            input_data = await self._get_step_input(
                step_number,
                pipeline_run_id=str(run.id),
                session=step_session,
            )

            logger.info("Running step %s: %s", step_number, STEP_NAMES[step_number])
            result = await service.run(input_data)

            if not result.success:
                raise RuntimeError(
                    f"Step {step_number} ({STEP_NAMES[step_number]}) failed: {result.error}"
                )

            return execution

    async def _get_step_service(
        self,
        step_number: int,
        execution: StepExecution,
        session: AsyncSession | None = None,
    ) -> BaseStepService | None:
        """Get the appropriate step service instance."""
        # Step service registry
        step_services = {
            0: Step00SetupService,
            1: Step01BrandService,
            2: Step02SeedsService,
            3: Step03ExpansionService,
            4: Step04MetricsService,
            5: Step05IntentService,
            6: Step06ClusteringService,
            7: Step07PrioritizationService,
            8: Step08SerpValidationService,
            # Steps 9-11 are optional and not yet implemented
            # 9: Step09InventoryService,
            # 10: Step10CannibalizationService,
            # 11: Step11LinkingService,
            12: Step12BriefService,
            13: Step13TemplatesService,
            14: Step14ArticleWriterService,
        }

        service_class = step_services.get(step_number)
        if service_class is None:
            return None

        return service_class(
            session=session or self.session,
            project_id=self.project_id,
            execution=execution,
        )

    async def _get_step_input(
        self,
        step_number: int,
        pipeline_run_id: str,
        session: AsyncSession | None = None,
    ) -> Any:
        """Get input data for a step from previous step outputs."""
        _session = session or self.session
        # Load project
        result = await _session.execute(select(Project).where(Project.id == self.project_id))
        project = result.scalar_one_or_none()

        if not project:
            raise ValueError(f"Project not found: {self.project_id}")

        # Step-specific input data
        step_inputs = {
            0: SetupInput(project_id=self.project_id),
            1: BrandInput(project_id=self.project_id),
            2: SeedsInput(project_id=self.project_id),
            3: ExpansionInput(project_id=self.project_id),
            4: MetricsInput(project_id=self.project_id),
            5: IntentInput(project_id=self.project_id),
            6: ClusteringInput(project_id=self.project_id),
            7: PrioritizationInput(project_id=self.project_id),
            8: SerpValidationInput(project_id=self.project_id),
            # Steps 9-11 would have their own input types
            12: BriefInput(project_id=self.project_id),
            13: TemplatesInput(project_id=self.project_id),
            14: ArticleWriterInput(project_id=self.project_id),
        }

        base_input = step_inputs.get(step_number, {"project_id": self.project_id})
        run_result = await _session.execute(
            select(PipelineRun).where(PipelineRun.id == pipeline_run_id)
        )
        run = run_result.scalar_one_or_none()
        if not run or not isinstance(run.steps_config, dict):
            return base_input

        step_inputs_cfg = run.steps_config.get("step_inputs")
        if not isinstance(step_inputs_cfg, dict):
            return base_input

        override = step_inputs_cfg.get(str(step_number)) or step_inputs_cfg.get(step_number)
        if not isinstance(override, dict) or not override:
            return base_input

        if isinstance(base_input, dict):
            return {**base_input, **override}

        try:
            merged = {**base_input.__dict__, **override}
            return type(base_input)(**merged)
        except Exception:
            logger.warning(
                "Invalid step input override ignored",
                extra={
                    "project_id": self.project_id,
                    "step_number": step_number,
                    "override_keys": sorted(override.keys()),
                },
            )
            return base_input

    async def _update_run_status(self, run_id: Any, **updates: Any) -> None:
        """Update PipelineRun status using a fresh DB session."""
        async with get_session_context() as session:
            result = await session.execute(select(PipelineRun).where(PipelineRun.id == run_id))
            run = result.scalar_one()
            run.patch(session, PipelineRunPatchDTO.from_partial(updates))

    async def _create_skipped_execution(
        self,
        run: PipelineRun,
        step_number: int,
        reason: str = "Skipped by configuration",
    ) -> StepExecution:
        """Create a skipped step execution record."""
        async with get_session_context() as session:
            execution = StepExecution.create(
                session,
                StepExecutionCreateDTO(
                    pipeline_run_id=str(run.id),
                    step_number=step_number,
                    step_name=STEP_NAMES[step_number],
                    status="skipped",
                    progress_message=reason,
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                ),
            )
        return execution

    async def _get_or_create_run(self) -> PipelineRun:
        """Get existing run or create a new one for single step execution."""
        if self._current_run:
            return self._current_run

        run = PipelineRun.create(
            self.session,
            PipelineRunCreateDTO(
                project_id=self.project_id,
                status="running",
                started_at=datetime.now(timezone.utc),
            ),
        )
        await self.session.flush()
        self._current_run = run
        return run

    async def _get_last_completed_step(self, run: PipelineRun) -> int | None:
        """Get the last completed step number for a run."""
        result = await self.session.execute(
            select(StepExecution)
            .where(
                StepExecution.pipeline_run_id == run.id,
                StepExecution.status == "completed",
            )
            .order_by(StepExecution.step_number.desc())
            .limit(1)
        )
        execution = result.scalar_one_or_none()
        return execution.step_number if execution else None

    async def _count_completed_steps(self, run_id: Any) -> int:
        """Count completed steps for a run."""
        result = await self.session.execute(
            select(func.count())
            .select_from(StepExecution)
            .where(
                StepExecution.pipeline_run_id == run_id,
                StepExecution.status == "completed",
            )
        )
        count = result.scalar_one()
        return int(count or 0)

    async def _set_step_skipped_task_state(self, task_id: str, step_num: int) -> None:
        """Set task stage when a step is skipped."""
        await self.task_manager.set_task_state(
            task_id=task_id,
            status="running",
            stage=f"Skipped step {step_num}: {STEP_NAMES.get(step_num, f'step_{step_num}')}",
            current_step=step_num,
            current_step_name=STEP_NAMES.get(step_num),
            error_message=None,
        )

    async def _has_existing_content(self) -> bool:
        """Check if the project has existing content to inventory."""
        # TODO: Implement actual check
        return False
