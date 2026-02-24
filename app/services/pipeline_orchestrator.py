"""Pipeline orchestrator facade for setup/discovery/content modules."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, cast

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import set_committed_value

from app.core.database import get_session_context, rollback_read_only_transaction
from app.core.exceptions import (
    PipelineAlreadyRunningError,
    PipelineDelayedResumeRequested,
    StepNotFoundError,
    StepPreconditionError,
)
from app.models.generated_dtos import (
    PipelineRunCreateDTO,
    StepExecutionCreateDTO,
)
from app.models.pipeline import PipelineRun, StepExecution
from app.models.project import Project
from app.repositories.pipeline_run_repository import PipelineRunRepository
from app.schemas.pipeline import ContentPipelineConfig, DiscoveryLoopConfig
from app.services.pipeline_task_manager import get_content_pipeline_task_manager
from app.services.pipelines import (
    CONTENT_DEFAULT_END_STEP,
    CONTENT_DEFAULT_START_STEP,
    CONTENT_LOCAL_STEP_DEPENDENCIES,
    CONTENT_LOCAL_STEP_NAMES,
    CONTENT_LOCAL_TO_INPUT,
    CONTENT_LOCAL_TO_SERVICE,
    DISCOVERY_DEFAULT_END_STEP,
    DISCOVERY_DEFAULT_START_STEP,
    DISCOVERY_LOCAL_STEP_DEPENDENCIES,
    DISCOVERY_LOCAL_STEP_NAMES,
    DISCOVERY_LOCAL_TO_INPUT,
    DISCOVERY_LOCAL_TO_SERVICE,
    SETUP_DEFAULT_END_STEP,
    SETUP_DEFAULT_START_STEP,
    SETUP_LOCAL_STEP_DEPENDENCIES,
    SETUP_LOCAL_STEP_NAMES,
    SETUP_LOCAL_TO_INPUT,
    SETUP_LOCAL_TO_SERVICE,
)
from app.services.pipelines.discovery.loop import (
    DISCOVERY_STEPS,
    AcceptedTopicState,
    DiscoveryLoopResult,
    DiscoveryLoopSupervisor,
)
from app.services.steps.base_step import BaseStepService
from app.services.task_manager import TaskManager

logger = logging.getLogger(__name__)

PipelineModule = Literal["setup", "discovery", "content"]


class PipelineOrchestrator:
    """Orchestrates execution of setup/discovery/content module runs."""

    def __init__(self, session: AsyncSession | None, project_id: str) -> None:
        self.session = session
        self.project_id = project_id
        self.pipeline_run_repository = PipelineRunRepository(project_id=project_id)
        self.task_manager = TaskManager()
        self._current_run: PipelineRun | None = None
        self._discovery_loop_state_key = "loop_state"
        self._discovery_loop_mode = "stepwise_discovery"

    @asynccontextmanager
    async def _active_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Yield caller session or a short-lived orchestrator session."""
        if self.session is not None:
            yield self.session
            return
        async with get_session_context(commit_on_exit=False) as fresh_session:
            yield fresh_session

    async def start_pipeline(
        self,
        *,
        start_step: int | None = None,
        end_step: int | None = None,
        skip_steps: list[int] | None = None,
        run_id: str | None = None,
        steps_config: dict[str, Any] | None = None,
        pipeline_module: PipelineModule | None = None,
    ) -> PipelineRun:
        """Start a pipeline run."""
        skip_steps = skip_steps or []
        run = await self._load_or_create_run(
            run_id=run_id,
            pipeline_module=pipeline_module,
            start_step=start_step,
            end_step=end_step,
            skip_steps=skip_steps,
            steps_config=steps_config,
        )
        run_module = self._as_pipeline_module(run.pipeline_module)

        await self._assert_can_start_module(
            pipeline_module=run_module,
            excluding_run_id=str(run.id),
        )

        module_cfg = self._module_config(run_module)
        effective_start = (
            start_step
            if start_step is not None
            else (
                run.start_step
                if run.start_step is not None
                else module_cfg["default_start"]
            )
        )
        effective_end = (
            end_step
            if end_step is not None
            else (run.end_step if run.end_step is not None else module_cfg["default_end"])
        )
        effective_skip = skip_steps if skip_steps else (run.skip_steps or [])
        effective_steps_config = self._build_effective_steps_config(
            pipeline_module=run_module,
            start_step=effective_start,
            end_step=effective_end,
            skip_steps=effective_skip,
            existing_config=run.steps_config,
            overrides=steps_config,
        )

        await self._patch_run(
            run,
            status="running",
            started_at=datetime.now(timezone.utc),
            completed_at=None,
            error_message=None,
            paused_at_step=None,
            start_step=effective_start,
            end_step=effective_end,
            skip_steps=effective_skip,
            steps_config=effective_steps_config,
        )
        self._current_run = run

        task_id = str(run.id)
        total_steps = len(
            [step for step in range(effective_start, effective_end + 1) if step not in effective_skip]
        )
        await self.task_manager.set_task_state(
            task_id=task_id,
            status="running",
            stage=f"{run_module.capitalize()} pipeline started",
            project_id=self.project_id,
            pipeline_module=run_module,
            source_topic_id=run.source_topic_id,
            current_step=effective_start,
            current_step_name=module_cfg["step_names"].get(effective_start),
            completed_steps=0,
            total_steps=total_steps,
            progress_percent=0.0,
            error_message=None,
        )

        try:
            if run_module == "discovery":
                discovery_result = await self._run_discovery_loop(run)
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
                        "Discovery completed "
                        f"({discovery_result.accepted_count}/{discovery_result.target_count} accepted topics)"
                    ),
                    pipeline_module=run_module,
                    source_topic_id=run.source_topic_id,
                    current_step=effective_end,
                    current_step_name=module_cfg["step_names"].get(effective_end),
                    progress_percent=100.0,
                    error_message=None,
                )
                return run

            await self._run_step_range(
                run=run,
                module=run_module,
                start_step=effective_start,
                end_step=effective_end,
                skip_steps=effective_skip,
                task_id=task_id,
            )
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
                stage=f"{run_module.capitalize()} pipeline completed",
                pipeline_module=run_module,
                source_topic_id=run.source_topic_id,
                current_step=effective_end,
                current_step_name=module_cfg["step_names"].get(effective_end),
                progress_percent=100.0,
                error_message=None,
            )

        except Exception as exc:
            paused_at_step = run.paused_at_step or effective_start
            await self._update_run_status(
                run.id,
                status="paused",
                paused_at_step=paused_at_step,
                error_message=str(exc),
            )
            run.status = "paused"
            run.paused_at_step = paused_at_step
            run.error_message = str(exc)
            await self.task_manager.set_task_state(
                task_id=task_id,
                status="paused",
                stage=f"{run_module.capitalize()} pipeline paused",
                pipeline_module=run_module,
                source_topic_id=run.source_topic_id,
                current_step=paused_at_step,
                current_step_name=module_cfg["step_names"].get(paused_at_step),
                error_message=str(exc),
            )

        return run

    async def resume_pipeline(
        self,
        run_id: str,
        pipeline_module: PipelineModule | None = None,
    ) -> PipelineRun:
        """Resume a paused pipeline run."""
        async with self._active_session() as session:
            result = await session.execute(
                select(PipelineRun).where(
                    PipelineRun.id == run_id,
                    PipelineRun.project_id == self.project_id,
                    PipelineRun.status == "paused",
                )
            )
        run = result.scalar_one_or_none()
        if run is None:
            raise ValueError("No paused pipeline found")
        run_module = self._as_pipeline_module(run.pipeline_module)
        if pipeline_module is not None and run_module != pipeline_module:
            raise ValueError("Paused run module does not match requested module")

        if run_module == "discovery":
            return await self.start_pipeline(
                run_id=run_id,
                pipeline_module=run_module,
                start_step=(
                    run.start_step
                    if run.start_step is not None
                    else DISCOVERY_DEFAULT_START_STEP
                ),
                end_step=(
                    run.end_step
                    if run.end_step is not None
                    else DISCOVERY_DEFAULT_END_STEP
                ),
                skip_steps=run.skip_steps or [],
                steps_config=run.steps_config,
            )

        module_cfg = self._module_config(run_module)
        last_completed = await self._get_last_completed_step(str(run.id))
        start_step = (
            (last_completed + 1)
            if last_completed is not None
            else (
                run.start_step
                if run.start_step is not None
                else module_cfg["default_start"]
            )
        )
        return await self.start_pipeline(
            run_id=run_id,
            pipeline_module=run_module,
            start_step=start_step,
            end_step=(
                run.end_step
                if run.end_step is not None
                else module_cfg["default_end"]
            ),
            skip_steps=run.skip_steps or [],
            steps_config=run.steps_config,
        )

    async def pause_pipeline(self, run_id: str) -> None:
        """Pause a running pipeline run."""
        async with self._active_session() as session:
            result = await session.execute(
                select(PipelineRun).where(
                    PipelineRun.id == run_id,
                    PipelineRun.project_id == self.project_id,
                    PipelineRun.status == "running",
                )
            )
        run = result.scalar_one_or_none()
        if run is None:
            raise ValueError("No running pipeline found")

        await self._patch_run(run, status="paused")
        await self.task_manager.set_task_state(
            task_id=str(run.id),
            status="paused",
            stage=f"{run.pipeline_module.capitalize()} pipeline paused",
            pipeline_module=run.pipeline_module,
            source_topic_id=run.source_topic_id,
        )

    async def run_single_step(
        self,
        step_number: int,
        pipeline_module: PipelineModule = "discovery",
    ) -> StepExecution:
        """Run one module-local step."""
        module_cfg = self._module_config(pipeline_module)
        if step_number not in module_cfg["step_names"]:
            raise StepNotFoundError(step_number)

        run = await self._get_or_create_single_step_run(pipeline_module)
        await self._verify_dependencies(run_id=str(run.id), module=pipeline_module, step_number=step_number)
        execution = await self._execute_step(run, pipeline_module, step_number)
        return execution

    async def run_queued_job_slice(
        self,
        *,
        run_id: str,
        pipeline_module: PipelineModule,
        job_kind: str,
    ) -> bool:
        """Execute one fair-scheduling slice for a queued run.

        Returns:
            True when more work remains and the job should be requeued.
        """
        async with self._active_session() as session:
            result = await session.execute(
                select(PipelineRun).where(
                    PipelineRun.id == run_id,
                    PipelineRun.project_id == self.project_id,
                )
            )
        run = result.scalar_one_or_none()
        if run is None:
            logger.warning(
                "Queued run missing, dropping job",
                extra={
                    "project_id": self.project_id,
                    "run_id": run_id,
                    "pipeline_module": pipeline_module,
                    "job_kind": job_kind,
                },
            )
            return False

        run_module = self._as_pipeline_module(run.pipeline_module)
        if run_module != pipeline_module:
            logger.warning(
                "Queued run module mismatch, dropping job",
                extra={
                    "project_id": self.project_id,
                    "run_id": run_id,
                    "expected_module": pipeline_module,
                    "actual_module": run_module,
                    "job_kind": job_kind,
                },
            )
            return False

        if run.status == "completed":
            return False

        if run.status == "paused" and job_kind != "resume":
            logger.info(
                "Ignoring stale non-resume job for paused run",
                extra={
                    "project_id": self.project_id,
                    "run_id": run_id,
                    "pipeline_module": pipeline_module,
                    "job_kind": job_kind,
                },
            )
            return False

        if run.status in {"pending", "paused"}:
            await self._assert_can_start_module(
                pipeline_module=run_module,
                excluding_run_id=str(run.id),
            )
            await self._mark_run_running_for_slice(
                run=run,
                resumed=(run.status == "paused"),
            )
        elif run.status != "running":
            logger.info(
                "Skipping queued run with terminal/non-runnable status",
                extra={
                    "project_id": self.project_id,
                    "run_id": run_id,
                    "pipeline_module": pipeline_module,
                    "status": run.status,
                },
            )
            return False

        run_id_str = str(run.id)
        run_source_topic_id = run.source_topic_id
        run_start_step = run.start_step
        await self._release_read_only_transaction()

        try:
            if run_module == "discovery":
                return await self._run_discovery_slice(run)
            return await self._run_standard_slice(run)
        except PipelineDelayedResumeRequested as exc:
            module_cfg = self._module_config(run_module)
            paused_at_step = module_cfg["default_start"]
            await self._update_run_status_with_retry(
                run_id=run_id_str,
                status="paused",
                paused_at_step=paused_at_step,
                error_message=None,
            )
            cooldown_minutes = max(1, int((exc.delay_seconds + 59) // 60))
            await self.task_manager.set_task_state(
                task_id=run_id_str,
                status="paused",
                stage=(
                    "Discovery cooldown active: auto-resume scheduled "
                    f"in ~{cooldown_minutes} minute(s)"
                ),
                pipeline_module=run_module,
                source_topic_id=run_source_topic_id,
                current_step=paused_at_step,
                current_step_name=module_cfg["step_names"].get(paused_at_step),
                error_message=None,
            )
            raise
        except Exception as exc:
            await self._rollback_failed_slice_session(
                run_id=run_id_str,
                pipeline_module=run_module,
                job_kind=job_kind,
            )
            logger.exception(
                "Queued pipeline slice failed; pausing run",
                extra={
                    "project_id": self.project_id,
                    "run_id": run_id,
                    "pipeline_module": run_module,
                    "job_kind": job_kind,
                },
            )
            module_cfg = self._module_config(run_module)
            run_state = run.__dict__ if isinstance(run.__dict__, dict) else {}
            paused_at_step: int | None = None
            for candidate in (
                run_state.get("paused_at_step"),
                run_state.get("start_step"),
                run_start_step,
                module_cfg["default_start"],
            ):
                if candidate is None:
                    continue
                try:
                    paused_at_step = int(candidate)
                    break
                except (TypeError, ValueError):
                    continue
            if paused_at_step is None:
                paused_at_step = module_cfg["default_start"]
            error_message = str(exc)
            await self._update_run_status_with_retry(
                run_id=run_id_str,
                status="paused",
                paused_at_step=paused_at_step,
                error_message=error_message,
            )
            await self.task_manager.set_task_state(
                task_id=run_id_str,
                status="paused",
                stage=f"{run_module.capitalize()} pipeline paused",
                pipeline_module=run_module,
                source_topic_id=run_source_topic_id,
                current_step=paused_at_step,
                current_step_name=module_cfg["step_names"].get(paused_at_step),
                error_message=error_message,
            )
            return False

    async def _rollback_failed_slice_session(
        self,
        *,
        run_id: str,
        pipeline_module: PipelineModule,
        job_kind: str,
    ) -> None:
        if self.session is None:
            # Short-lived sessions are already closed per operation.
            return
        in_transaction = getattr(self.session, "in_transaction", None)
        if callable(in_transaction):
            try:
                if not in_transaction():
                    return
            except Exception:
                pass
        try:
            await self.session.rollback()
        except Exception as exc:
            error_text = str(exc).lower()
            if "connection is closed" in error_text or "underlying connection is closed" in error_text:
                logger.warning(
                    "Skipping rollback after slice failure because session connection is already closed",
                    extra={
                        "project_id": self.project_id,
                        "run_id": run_id,
                        "pipeline_module": pipeline_module,
                        "job_kind": job_kind,
                    },
                )
                return
            logger.exception(
                "Failed to rollback session after slice failure",
                extra={
                    "project_id": self.project_id,
                    "run_id": run_id,
                    "pipeline_module": pipeline_module,
                    "job_kind": job_kind,
                },
            )

    async def _release_read_only_transaction(self, session: AsyncSession | None = None) -> None:
        target_session = session or self.session
        if target_session is None:
            return
        await rollback_read_only_transaction(
            target_session,
            context="pipeline_orchestrator",
        )

    async def _update_run_status_with_retry(
        self,
        *,
        run_id: Any,
        status: str,
        paused_at_step: int | None,
        error_message: str | None,
        attempts: int = 2,
    ) -> None:
        await self._update_run_status(
            run_id,
            attempts=attempts,
            status=status,
            paused_at_step=paused_at_step,
            error_message=error_message,
        )

    async def _mark_run_running_for_slice(
        self,
        *,
        run: PipelineRun,
        resumed: bool,
    ) -> None:
        module_cfg = self._module_config(run.pipeline_module)
        start_step = run.start_step if run.start_step is not None else module_cfg["default_start"]
        await self._patch_run(
            run,
            status="running",
            started_at=datetime.now(timezone.utc),
            completed_at=None,
            error_message=None,
        )
        stage = (
            f"{run.pipeline_module.capitalize()} pipeline resumed"
            if resumed
            else f"{run.pipeline_module.capitalize()} pipeline started"
        )
        await self.task_manager.set_task_state(
            task_id=str(run.id),
            status="running",
            stage=stage,
            project_id=self.project_id,
            pipeline_module=run.pipeline_module,
            source_topic_id=run.source_topic_id,
            current_step=start_step,
            current_step_name=module_cfg["step_names"].get(start_step),
            error_message=None,
        )

    async def _run_standard_slice(self, run: PipelineRun) -> bool:
        module = self._as_pipeline_module(run.pipeline_module)
        module_cfg = self._module_config(module)
        next_step = await self._next_standard_step(run=run, module=module)
        if next_step is None:
            await self._mark_run_completed(
                run=run,
                completion_stage=f"{module.capitalize()} pipeline completed",
            )
            return False

        skip_steps = run.skip_steps or []
        if next_step in skip_steps and next_step in module_cfg["optional_steps"]:
            await self._create_skipped_execution(run, module, next_step)
            await self._set_step_skipped_task_state(
                task_id=str(run.id),
                module=module,
                step_num=next_step,
            )
        else:
            await self._verify_dependencies(run_id=str(run.id), module=module, step_number=next_step)
            await self._release_read_only_transaction()
            execution = await self._execute_step(run, module, next_step)
            await self._patch_run(run, paused_at_step=next_step)
            if execution.status == "completed":
                await self.task_manager.mark_step_completed(
                    task_id=str(run.id),
                    step_number=next_step,
                    step_name=module_cfg["step_names"].get(next_step, f"step_{next_step}"),
                )
            elif execution.status == "skipped":
                await self._set_step_skipped_task_state(
                    task_id=str(run.id),
                    module=module,
                    step_num=next_step,
                )

        has_more = await self._next_standard_step(run=run, module=module) is not None
        if not has_more:
            await self._mark_run_completed(
                run=run,
                completion_stage=f"{module.capitalize()} pipeline completed",
            )
            return False
        return True

    async def _next_standard_step(
        self,
        *,
        run: PipelineRun,
        module: PipelineModule,
    ) -> int | None:
        module_cfg = self._module_config(module)
        start_step = run.start_step if run.start_step is not None else module_cfg["default_start"]
        end_step = run.end_step if run.end_step is not None else module_cfg["default_end"]
        skip_steps = run.skip_steps or []

        async with self._active_session() as session:
            completed_result = await session.execute(
                select(StepExecution.step_number).where(
                    StepExecution.pipeline_run_id == str(run.id),
                    StepExecution.status.in_(["completed", "skipped"]),
                )
            )
        completed_steps = {int(step_num) for step_num in completed_result.scalars().all()}

        for step_num in range(start_step, end_step + 1):
            if step_num in completed_steps:
                continue
            if step_num in skip_steps and step_num in module_cfg["optional_steps"]:
                return step_num
            return step_num
        return None

    async def _mark_run_completed(
        self,
        *,
        run: PipelineRun,
        completion_stage: str,
    ) -> None:
        module_cfg = self._module_config(run.pipeline_module)
        effective_end = run.end_step if run.end_step is not None else module_cfg["default_end"]
        now = datetime.now(timezone.utc)
        await self._update_run_status(
            run.id,
            status="completed",
            completed_at=now,
            paused_at_step=None,
            error_message=None,
        )
        run.status = "completed"
        run.completed_at = now
        run.paused_at_step = None
        run.error_message = None
        await self.task_manager.set_task_state(
            task_id=str(run.id),
            status="completed",
            stage=completion_stage,
            pipeline_module=run.pipeline_module,
            source_topic_id=run.source_topic_id,
            current_step=effective_end,
            current_step_name=module_cfg["step_names"].get(effective_end),
            progress_percent=100.0,
            error_message=None,
        )

    async def _run_discovery_slice(self, run: PipelineRun) -> bool:
        async with self._active_session() as loop_session:
            steps_config = dict(run.steps_config or {})
            supervisor = DiscoveryLoopSupervisor(
                session=loop_session,
                project_id=self.project_id,
                run=run,
                task_manager=self.task_manager,
            )
            discovery_cfg = DiscoveryLoopConfig.model_validate(steps_config.get("discovery") or {})
            content_cfg = ContentPipelineConfig.model_validate(steps_config.get("content") or {})

            state = self._get_discovery_loop_state(steps_config)
            if state is None:
                state = await self._initialize_discovery_loop_state(
                    run=run,
                    steps_config=steps_config,
                    supervisor=supervisor,
                    discovery_cfg=discovery_cfg,
                )

            current_iteration = int(state.get("current_iteration", 1))
            step_cursor = int(state.get("step_cursor", 0))
            task_id = str(run.id)

            if step_cursor < len(DISCOVERY_STEPS):
                step_num = DISCOVERY_STEPS[step_cursor]
                execution: StepExecution
                module_cfg = self._module_config("discovery")
                skip_steps = run.skip_steps or []
                if step_num in skip_steps and step_num in module_cfg["optional_steps"]:
                    execution = await self._create_skipped_execution(run, "discovery", step_num)
                    await self._set_step_skipped_task_state(
                        task_id=task_id,
                        module="discovery",
                        step_num=step_num,
                    )
                else:
                    await self._verify_dependencies(
                        run_id=task_id,
                        module="discovery",
                        step_number=step_num,
                    )
                    await self._release_read_only_transaction(loop_session)
                    execution = await self._execute_step(run, "discovery", step_num)
                    await self._patch_run(run, paused_at_step=step_num)
                    if execution.status == "completed":
                        await self.task_manager.mark_step_completed(
                            task_id=task_id,
                            step_number=step_num,
                            step_name=module_cfg["step_names"].get(step_num, f"step_{step_num}"),
                        )
                    elif execution.status == "skipped":
                        await self._set_step_skipped_task_state(
                            task_id=task_id,
                            module="discovery",
                            step_num=step_num,
                        )

                summaries_raw = state.get("step_summaries")
                summaries_in = summaries_raw if isinstance(summaries_raw, dict) else {}
                summaries_typed: dict[int, dict[str, Any]] = {}
                for key, value in summaries_in.items():
                    if not isinstance(value, dict):
                        continue
                    try:
                        summaries_typed[int(key)] = value
                    except (TypeError, ValueError):
                        continue
                summaries_typed = supervisor._collect_iteration_step_summaries(
                    summaries_typed,
                    step_num=step_num,
                    execution=execution,
                )
                state["step_cursor"] = step_cursor + 1
                state["step_summaries"] = {str(key): value for key, value in summaries_typed.items()}
                await self._save_discovery_loop_state(
                    run=run,
                    steps_config=steps_config,
                    state=state,
                )
                return True

            decisions = await supervisor._evaluate_topic_decisions(
                iteration_index=current_iteration,
                discovery=discovery_cfg,
            )
            await supervisor._persist_snapshots(current_iteration, decisions)

            summaries_state = state.get("step_summaries")
            summaries_raw = summaries_state if isinstance(summaries_state, dict) else {}
            step_summaries: dict[int, dict[str, Any]] = {}
            for key, value in summaries_raw.items():
                if not isinstance(value, dict):
                    continue
                try:
                    step_summaries[int(key)] = value
                except (TypeError, ValueError):
                    continue
            await supervisor._persist_iteration_learnings(
                iteration_index=current_iteration,
                decisions=decisions,
                step_summaries=step_summaries,
            )

            accepted_pool = self._deserialize_accepted_topics(state.get("accepted_topics"))
            accepted_pool = supervisor._merge_accepted_topics(
                current_pool=accepted_pool,
                decisions=decisions,
            )
            accepted_topic_ids = supervisor._collect_selected_topic_ids(accepted_pool)
            accepted_topic_names = supervisor._collect_selected_topic_names(accepted_pool)

            steps_config["selected_topic_ids"] = accepted_topic_ids
            steps_config["selected_topic_names"] = accepted_topic_names
            steps_config["accepted_topic_count"] = len(accepted_pool)
            steps_config["iteration_index"] = current_iteration
            await self._persist_run_steps_config(run=run, steps_config=steps_config)

            await self._dispatch_new_topics_from_discovery(
                run=run,
                accepted_topic_ids=accepted_topic_ids,
                content_config=content_cfg,
            )

            accepted_count = len(accepted_pool)
            target_count = int(state.get("target_count", 1))
            if accepted_count >= target_count:
                await self._mark_run_completed(
                    run=run,
                    completion_stage=(
                        "Discovery completed "
                        f"({accepted_count}/{target_count} accepted topics)"
                    ),
                )
                return False

            if current_iteration >= discovery_cfg.max_iterations:
                if discovery_cfg.auto_resume_on_exhaustion:
                    immutable_excludes_raw = state.get("immutable_excludes")
                    immutable_excludes = (
                        {str(item).strip().lower() for item in immutable_excludes_raw if str(item).strip()}
                        if isinstance(immutable_excludes_raw, list)
                        else set()
                    )
                    dynamic_excludes = supervisor._next_dynamic_excludes(
                        current_dynamic_excludes=[
                            str(item)
                            for item in state.get("dynamic_excludes", [])
                            if str(item).strip()
                        ],
                        decisions=decisions,
                        immutable_excludes=immutable_excludes,
                    )
                    base_strategy_payload_raw = state.get("base_strategy_payload")
                    base_strategy_payload = (
                        dict(base_strategy_payload_raw)
                        if isinstance(base_strategy_payload_raw, dict)
                        else {}
                    )
                    restart_strategy_payload = supervisor._build_iteration_strategy_payload(
                        base_strategy_payload=base_strategy_payload,
                        iteration=1,
                        dynamic_excludes=dynamic_excludes,
                    )

                    cooldown_delta = timedelta(minutes=discovery_cfg.exhaustion_cooldown_minutes)
                    cooldown_until = datetime.now(timezone.utc) + cooldown_delta
                    steps_config["strategy"] = restart_strategy_payload
                    steps_config["iteration_index"] = 1
                    state["current_iteration"] = 1
                    state["step_cursor"] = 0
                    state["dynamic_excludes"] = dynamic_excludes
                    state["accepted_topics"] = self._serialize_accepted_topics(accepted_pool)
                    state["step_summaries"] = {}
                    state["cooldown_until"] = cooldown_until.isoformat()
                    state["last_exhausted_iteration"] = current_iteration
                    await self._save_discovery_loop_state(
                        run=run,
                        steps_config=steps_config,
                        state=state,
                    )
                    raise PipelineDelayedResumeRequested(
                        delay_seconds=cooldown_delta.total_seconds(),
                        reason="discovery_max_iterations_exhausted",
                    )
                raise RuntimeError(
                    "Insufficient accepted topics after discovery loop "
                    f"({accepted_count}/{target_count} accepted across {current_iteration} iterations). "
                    "Try broadening topic scope, adding include_topics, or lowering difficulty constraints."
                )

            immutable_excludes_raw = state.get("immutable_excludes")
            immutable_excludes = (
                {str(item).strip().lower() for item in immutable_excludes_raw if str(item).strip()}
                if isinstance(immutable_excludes_raw, list)
                else set()
            )
            dynamic_excludes = supervisor._next_dynamic_excludes(
                current_dynamic_excludes=[
                    str(item)
                    for item in state.get("dynamic_excludes", [])
                    if str(item).strip()
                ],
                decisions=decisions,
                immutable_excludes=immutable_excludes,
            )

            base_strategy_payload_raw = state.get("base_strategy_payload")
            base_strategy_payload = (
                dict(base_strategy_payload_raw)
                if isinstance(base_strategy_payload_raw, dict)
                else {}
            )
            next_iteration = current_iteration + 1
            strategy_payload = supervisor._build_iteration_strategy_payload(
                base_strategy_payload=base_strategy_payload,
                iteration=next_iteration,
                dynamic_excludes=dynamic_excludes,
            )

            steps_config["strategy"] = strategy_payload
            steps_config["iteration_index"] = next_iteration
            state["current_iteration"] = next_iteration
            state["step_cursor"] = 0
            state["dynamic_excludes"] = dynamic_excludes
            state["accepted_topics"] = self._serialize_accepted_topics(accepted_pool)
            state["step_summaries"] = {}
            await self._save_discovery_loop_state(
                run=run,
                steps_config=steps_config,
                state=state,
            )

            await self.task_manager.set_task_state(
                task_id=task_id,
                status="running",
                stage=f"Discovery loop iteration {next_iteration}/{discovery_cfg.max_iterations}",
                current_step=1,
                current_step_name="seed_topics",
                error_message=None,
            )
            return True

    async def _initialize_discovery_loop_state(
        self,
        *,
        run: PipelineRun,
        steps_config: dict[str, Any],
        supervisor: DiscoveryLoopSupervisor,
        discovery_cfg: DiscoveryLoopConfig,
    ) -> dict[str, Any]:
        base_strategy_payload = dict(steps_config.get("strategy") or {})
        immutable_excludes = [
            str(topic).strip().lower()
            for topic in base_strategy_payload.get("exclude_topics", [])
            if str(topic).strip()
        ]
        target_count = await supervisor._resolve_target_count(discovery_cfg, base_strategy_payload)
        strategy_payload = supervisor._build_iteration_strategy_payload(
            base_strategy_payload=base_strategy_payload,
            iteration=1,
            dynamic_excludes=[],
        )

        steps_config["strategy"] = strategy_payload
        steps_config["iteration_index"] = 1
        steps_config.setdefault("selected_topic_ids", [])
        steps_config.setdefault("selected_topic_names", [])
        steps_config.setdefault("accepted_topic_count", 0)

        state: dict[str, Any] = {
            "mode": self._discovery_loop_mode,
            "target_count": target_count,
            "current_iteration": 1,
            "step_cursor": 0,
            "dynamic_excludes": [],
            "immutable_excludes": immutable_excludes,
            "accepted_topics": [],
            "step_summaries": {},
            "base_strategy_payload": base_strategy_payload,
        }
        await self._save_discovery_loop_state(
            run=run,
            steps_config=steps_config,
            state=state,
        )
        await self.task_manager.set_task_state(
            task_id=str(run.id),
            status="running",
            stage=(
                f"Discovery loop started: target {target_count} accepted topics "
                f"in <= {discovery_cfg.max_iterations} iterations"
            ),
            project_id=self.project_id,
            current_step=1,
            current_step_name="seed_topics",
            error_message=None,
        )
        return state

    async def _save_discovery_loop_state(
        self,
        *,
        run: PipelineRun,
        steps_config: dict[str, Any],
        state: dict[str, Any],
    ) -> None:
        normalized = dict(steps_config)
        normalized[self._discovery_loop_state_key] = state
        await self._persist_run_steps_config(run=run, steps_config=normalized)

    def _get_discovery_loop_state(self, steps_config: dict[str, Any]) -> dict[str, Any] | None:
        raw = steps_config.get(self._discovery_loop_state_key)
        if not isinstance(raw, dict):
            return None
        if raw.get("mode") != self._discovery_loop_mode:
            return None
        return dict(raw)

    async def _persist_run_steps_config(
        self,
        *,
        run: PipelineRun,
        steps_config: dict[str, Any],
    ) -> None:
        await self._patch_run(run, steps_config=steps_config)

    def _serialize_accepted_topics(
        self,
        topics: dict[str, AcceptedTopicState],
    ) -> list[dict[str, str | None]]:
        serialized: list[dict[str, str | None]] = []
        for key, state in topics.items():
            serialized.append(
                {
                    "key": key,
                    "topic_name": state.topic_name,
                    "source_topic_id": state.source_topic_id,
                }
            )
        return serialized

    def _deserialize_accepted_topics(self, raw: Any) -> dict[str, AcceptedTopicState]:
        if not isinstance(raw, list):
            return {}
        parsed: dict[str, AcceptedTopicState] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or "").strip().lower()
            topic_name = str(item.get("topic_name") or "").strip()
            source_topic_id_raw = item.get("source_topic_id")
            source_topic_id = (
                str(source_topic_id_raw).strip()
                if source_topic_id_raw is not None and str(source_topic_id_raw).strip()
                else None
            )
            if not key:
                continue
            parsed[key] = AcceptedTopicState(
                topic_name=topic_name,
                source_topic_id=source_topic_id,
            )
        return parsed

    async def _load_or_create_run(
        self,
        *,
        run_id: str | None,
        pipeline_module: PipelineModule | None,
        start_step: int | None,
        end_step: int | None,
        skip_steps: list[int],
        steps_config: dict[str, Any] | None,
    ) -> PipelineRun:
        if run_id is not None:
            async with self._active_session() as session:
                result = await session.execute(
                    select(PipelineRun).where(
                        PipelineRun.id == run_id,
                        PipelineRun.project_id == self.project_id,
                    )
                )
            run = result.scalar_one_or_none()
            if run is None:
                raise ValueError(f"Pipeline run not found: {run_id}")
            if pipeline_module is not None and run.pipeline_module != pipeline_module:
                raise ValueError("Run module does not match requested module")
            return run

        if pipeline_module is None:
            raise ValueError("pipeline_module is required when creating a new run")
        module_cfg = self._module_config(pipeline_module)
        run_start = start_step if start_step is not None else module_cfg["default_start"]
        run_end = end_step if end_step is not None else module_cfg["default_end"]
        config = self._build_effective_steps_config(
            pipeline_module=pipeline_module,
            start_step=run_start,
            end_step=run_end,
            skip_steps=skip_steps,
            existing_config=steps_config,
        )
        async with self._active_session() as session:
            run = PipelineRun.create(
                session,
                PipelineRunCreateDTO(
                    project_id=self.project_id,
                    pipeline_module=pipeline_module,
                    status="pending",
                    start_step=run_start,
                    end_step=run_end,
                    skip_steps=skip_steps,
                    steps_config=config,
                ),
            )
            await session.commit()
        return run

    async def _run_step_range(
        self,
        *,
        run: PipelineRun,
        module: PipelineModule,
        start_step: int,
        end_step: int,
        skip_steps: list[int],
        task_id: str,
    ) -> None:
        module_cfg = self._module_config(module)
        for step_num in range(start_step, end_step + 1):
            if step_num in skip_steps and step_num in module_cfg["optional_steps"]:
                await self._create_skipped_execution(run, module, step_num)
                await self._set_step_skipped_task_state(
                    task_id=task_id,
                    module=module,
                    step_num=step_num,
                )
                continue

            await self._verify_dependencies(run_id=str(run.id), module=module, step_number=step_num)
            execution = await self._execute_step(run, module, step_num)
            run.paused_at_step = step_num
            if execution.status == "completed":
                await self.task_manager.mark_step_completed(
                    task_id=task_id,
                    step_number=step_num,
                    step_name=module_cfg["step_names"].get(step_num, f"step_{step_num}"),
                )
            elif execution.status == "skipped":
                await self._set_step_skipped_task_state(task_id=task_id, module=module, step_num=step_num)

    async def _run_discovery_loop(self, run: PipelineRun) -> DiscoveryLoopResult:
        async with self._active_session() as session:
            supervisor = DiscoveryLoopSupervisor(
                session=session,
                project_id=self.project_id,
                run=run,
                task_manager=self.task_manager,
            )
            return await supervisor.run_loop(
                execute_step=lambda _run, step_num: self._execute_step(_run, "discovery", step_num),
                mark_step_completed=self.task_manager.mark_step_completed,
                dispatch_accepted_topics=lambda accepted_topic_ids, content_cfg: self._dispatch_new_topics_from_discovery(
                    run=run,
                    accepted_topic_ids=accepted_topic_ids,
                    content_config=content_cfg,
                ),
            )

    async def _dispatch_new_topics_from_discovery(
        self,
        *,
        run: PipelineRun,
        accepted_topic_ids: list[str],
        content_config: ContentPipelineConfig,
    ) -> None:
        if not accepted_topic_ids:
            return
        if not self._discovery_auto_dispatch_enabled(run.steps_config):
            return

        async with self._active_session() as session:
            existing_result = await session.execute(
                select(PipelineRun.source_topic_id).where(
                    PipelineRun.parent_run_id == str(run.id),
                    PipelineRun.pipeline_module == "content",
                    PipelineRun.source_topic_id.is_not(None),
                )
            )
            dispatched_topic_ids = {str(topic_id) for topic_id in existing_result.scalars() if topic_id}
        newly_accepted_topic_ids = [topic_id for topic_id in accepted_topic_ids if topic_id not in dispatched_topic_ids]
        if not newly_accepted_topic_ids:
            return

        queue = get_content_pipeline_task_manager()
        for topic_id in newly_accepted_topic_ids:
            step_inputs = {
                "1": {
                    "topic_ids": [topic_id],
                    "max_briefs": 1,
                    "posts_per_week": content_config.posts_per_week,
                    "preferred_weekdays": content_config.preferred_weekdays,
                    "min_lead_days": content_config.min_lead_days,
                    "publication_start_date": content_config.publication_start_date,
                    "use_llm_timing_hints": content_config.use_llm_timing_hints,
                    "llm_timing_flex_days": content_config.llm_timing_flex_days,
                    "include_zero_data_topics": content_config.include_zero_data_topics,
                    "zero_data_topic_share": content_config.zero_data_topic_share,
                    "zero_data_fit_score_min": content_config.zero_data_fit_score_min,
                }
            }
            steps_config = self._build_effective_steps_config(
                pipeline_module="content",
                start_step=CONTENT_DEFAULT_START_STEP,
                end_step=CONTENT_DEFAULT_END_STEP,
                skip_steps=[],
                existing_config={
                    "strategy": (run.steps_config or {}).get("strategy"),
                    "content": content_config.model_dump(),
                    "step_inputs": step_inputs,
                },
            )
            async with self._active_session() as session:
                content_run = PipelineRun.create(
                    session,
                    PipelineRunCreateDTO(
                        project_id=self.project_id,
                        pipeline_module="content",
                        parent_run_id=str(run.id),
                        source_topic_id=topic_id,
                        status="pending",
                        start_step=CONTENT_DEFAULT_START_STEP,
                        end_step=CONTENT_DEFAULT_END_STEP,
                        skip_steps=[],
                        steps_config=steps_config,
                    ),
                )
                await session.commit()
            await self.task_manager.set_task_state(
                task_id=str(content_run.id),
                status="queued",
                stage="Queued content task from discovery acceptance",
                project_id=self.project_id,
                pipeline_module="content",
                source_topic_id=topic_id,
                current_step=CONTENT_DEFAULT_START_STEP,
                current_step_name=CONTENT_LOCAL_STEP_NAMES.get(CONTENT_DEFAULT_START_STEP),
                completed_steps=0,
                total_steps=CONTENT_DEFAULT_END_STEP,
                progress_percent=0.0,
                error_message=None,
            )
            await queue.enqueue_start(
                project_id=self.project_id,
                run_id=str(content_run.id),
            )

    async def _verify_dependencies(
        self,
        *,
        run_id: str,
        module: PipelineModule,
        step_number: int,
    ) -> None:
        deps = self._module_config(module)["dependencies"].get(step_number, [])
        for dep in deps:
            if not await self._is_step_completed(run_id=run_id, step_number=dep):
                step_names = self._module_config(module)["step_names"]
                raise StepPreconditionError(
                    step_number,
                    f"Step {dep} ({step_names.get(dep, dep)}) must be completed first",
                )

    async def _is_step_completed(self, *, run_id: str, step_number: int) -> bool:
        async with self._active_session() as session:
            result = await session.execute(
                select(StepExecution.id).where(
                    StepExecution.pipeline_run_id == run_id,
                    StepExecution.step_number == step_number,
                    StepExecution.status == "completed",
                ).limit(1)
            )
        return result.scalar() is not None

    async def _execute_step(
        self,
        run: PipelineRun,
        module: PipelineModule,
        step_number: int,
    ) -> StepExecution:
        async with get_session_context() as step_session:
            step_names = self._module_config(module)["step_names"]
            execution = StepExecution.create(
                step_session,
                StepExecutionCreateDTO(
                    pipeline_run_id=str(run.id),
                    step_number=step_number,
                    step_name=step_names[step_number],
                    status="pending",
                ),
            )
            await step_session.flush()

            service = await self._get_step_service(
                module=module,
                step_number=step_number,
                execution=execution,
                session=step_session,
            )
            if service is None:
                execution.status = "skipped"
                execution.progress_message = "Step not implemented"
                await step_session.commit()
                return execution

            input_data = await self._get_step_input(
                module=module,
                step_number=step_number,
                pipeline_run_id=str(run.id),
                session=step_session,
            )
            result = await service.run(input_data)
            if not result.success:
                raise RuntimeError(
                    f"Step {step_number} ({step_names[step_number]}) failed: {result.error}"
                )
            return execution

    async def _get_step_service(
        self,
        *,
        module: PipelineModule,
        step_number: int,
        execution: StepExecution,
        session: AsyncSession | None = None,
    ) -> BaseStepService | None:
        service_map = self._module_config(module)["service_map"]
        service_class = service_map.get(step_number)
        if service_class is None:
            return None
        if session is None and self.session is None:
            raise RuntimeError("No active session available for step service construction")
        return service_class(
            session=session or self.session,
            project_id=self.project_id,
            execution=execution,
        )

    async def _get_step_input(
        self,
        *,
        module: PipelineModule,
        step_number: int,
        pipeline_run_id: str,
        session: AsyncSession | None = None,
    ) -> Any:
        _session = session or self.session
        if _session is None:
            raise RuntimeError("No active session available for step input loading")
        project_result = await _session.execute(select(Project).where(Project.id == self.project_id))
        if project_result.scalar_one_or_none() is None:
            raise ValueError(f"Project not found: {self.project_id}")

        input_map = self._module_config(module)["input_map"]
        input_cls = input_map.get(step_number)
        base_input: Any = input_cls(project_id=self.project_id) if input_cls else {"project_id": self.project_id}

        run_result = await _session.execute(select(PipelineRun).where(PipelineRun.id == pipeline_run_id))
        run = run_result.scalar_one_or_none()
        if run is None or not isinstance(run.steps_config, dict):
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
            return base_input

    async def _patch_run(self, run: PipelineRun, *, attempts: int = 3, **updates: Any) -> None:
        await self.pipeline_run_repository.patch(
            run_id=str(run.id),
            updates=updates,
            attempts=attempts,
        )
        for field, value in updates.items():
            try:
                set_committed_value(run, field, value)
            except Exception:
                setattr(run, field, value)

    async def _update_run_status(
        self,
        run_id: Any,
        *,
        attempts: int = 3,
        **updates: Any,
    ) -> None:
        await self.pipeline_run_repository.patch(
            run_id=str(run_id),
            updates=updates,
            attempts=attempts,
        )

    async def _create_skipped_execution(
        self,
        run: PipelineRun,
        module: PipelineModule,
        step_number: int,
        reason: str = "Skipped by configuration",
    ) -> StepExecution:
        async with get_session_context() as fresh_session:
            execution = StepExecution.create(
                fresh_session,
                StepExecutionCreateDTO(
                    pipeline_run_id=str(run.id),
                    step_number=step_number,
                    step_name=self._module_config(module)["step_names"][step_number],
                    status="skipped",
                    progress_message=reason,
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                ),
            )
        return execution

    async def _get_or_create_single_step_run(self, module: PipelineModule) -> PipelineRun:
        if self._current_run and self._current_run.pipeline_module == module:
            return self._current_run

        module_cfg = self._module_config(module)
        async with self._active_session() as session:
            run = PipelineRun.create(
                session,
                PipelineRunCreateDTO(
                    project_id=self.project_id,
                    pipeline_module=module,
                    status="running",
                    started_at=datetime.now(timezone.utc),
                    start_step=module_cfg["default_start"],
                    end_step=module_cfg["default_end"],
                    skip_steps=[],
                    steps_config=self._build_effective_steps_config(
                        pipeline_module=module,
                        start_step=module_cfg["default_start"],
                        end_step=module_cfg["default_end"],
                        skip_steps=[],
                        existing_config=None,
                    ),
                ),
            )
            await session.commit()
        self._current_run = run
        return run

    async def _get_last_completed_step(self, run_id: str) -> int | None:
        async with self._active_session() as session:
            result = await session.execute(
                select(StepExecution)
                .where(
                    StepExecution.pipeline_run_id == run_id,
                    StepExecution.status == "completed",
                )
                .order_by(StepExecution.step_number.desc())
                .limit(1)
            )
        execution = result.scalar_one_or_none()
        return execution.step_number if execution else None

    async def _count_completed_steps(self, run_id: str) -> int:
        async with self._active_session() as session:
            result = await session.execute(
                select(func.count())
                .select_from(StepExecution)
                .where(
                    StepExecution.pipeline_run_id == run_id,
                    StepExecution.status == "completed",
                )
            )
        return int(result.scalar_one() or 0)

    async def _set_step_skipped_task_state(
        self,
        *,
        task_id: str,
        module: PipelineModule,
        step_num: int,
    ) -> None:
        step_name = self._module_config(module)["step_names"].get(step_num)
        await self.task_manager.set_task_state(
            task_id=task_id,
            status="running",
            stage=f"Skipped step {step_num}: {step_name or f'step_{step_num}'}",
            pipeline_module=module,
            current_step=step_num,
            current_step_name=step_name,
            error_message=None,
        )

    async def _assert_can_start_module(
        self,
        *,
        pipeline_module: str,
        excluding_run_id: str | None = None,
    ) -> None:
        query = (
            select(PipelineRun.id)
            .where(
                PipelineRun.project_id == self.project_id,
                PipelineRun.pipeline_module == pipeline_module,
                PipelineRun.status == "running",
            )
            .limit(1)
        )
        if excluding_run_id:
            query = query.where(PipelineRun.id != excluding_run_id)
        async with self._active_session() as session:
            result = await session.execute(query)
        if result.scalar_one_or_none() is not None:
            raise PipelineAlreadyRunningError(self.project_id)

    def _as_pipeline_module(self, module: str) -> PipelineModule:
        if module not in {"setup", "discovery", "content"}:
            raise ValueError(f"Unknown pipeline module: {module}")
        return cast(PipelineModule, module)

    def _module_config(self, module: str) -> dict[str, Any]:
        if module == "setup":
            return {
                "default_start": SETUP_DEFAULT_START_STEP,
                "default_end": SETUP_DEFAULT_END_STEP,
                "step_names": SETUP_LOCAL_STEP_NAMES,
                "dependencies": SETUP_LOCAL_STEP_DEPENDENCIES,
                "service_map": SETUP_LOCAL_TO_SERVICE,
                "input_map": SETUP_LOCAL_TO_INPUT,
                "optional_steps": set(),
            }
        if module == "discovery":
            return {
                "default_start": DISCOVERY_DEFAULT_START_STEP,
                "default_end": DISCOVERY_DEFAULT_END_STEP,
                "step_names": DISCOVERY_LOCAL_STEP_NAMES,
                "dependencies": DISCOVERY_LOCAL_STEP_DEPENDENCIES,
                "service_map": DISCOVERY_LOCAL_TO_SERVICE,
                "input_map": DISCOVERY_LOCAL_TO_INPUT,
                "optional_steps": {7},
            }
        if module == "content":
            return {
                "default_start": CONTENT_DEFAULT_START_STEP,
                "default_end": CONTENT_DEFAULT_END_STEP,
                "step_names": CONTENT_LOCAL_STEP_NAMES,
                "dependencies": CONTENT_LOCAL_STEP_DEPENDENCIES,
                "service_map": CONTENT_LOCAL_TO_SERVICE,
                "input_map": CONTENT_LOCAL_TO_INPUT,
                "optional_steps": set(),
            }
        raise ValueError(f"Unknown pipeline module: {module}")

    def _build_effective_steps_config(
        self,
        *,
        pipeline_module: PipelineModule,
        start_step: int,
        end_step: int,
        skip_steps: list[int],
        existing_config: dict[str, Any] | None,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = dict(existing_config or {})
        if overrides:
            config.update(overrides)
        config["pipeline_module"] = pipeline_module
        config["start"] = start_step
        config["end"] = end_step
        config["skip"] = skip_steps
        config.setdefault("iteration_index", 0)
        config.setdefault("selected_topic_ids", [])
        config.setdefault("step_inputs", {})
        return config

    def _discovery_auto_dispatch_enabled(self, steps_config: dict[str, Any] | None) -> bool:
        if not steps_config:
            return True
        discovery_cfg = steps_config.get("discovery")
        if not isinstance(discovery_cfg, dict):
            return True
        return bool(discovery_cfg.get("auto_dispatch_content_tasks", True))
