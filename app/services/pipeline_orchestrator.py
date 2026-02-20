"""Pipeline orchestrator facade for setup/discovery/content modules."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Literal

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
from app.schemas.pipeline import ContentPipelineConfig
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
from app.services.pipelines.discovery.loop import DiscoveryLoopResult, DiscoveryLoopSupervisor
from app.services.steps.base_step import BaseStepService
from app.services.task_manager import TaskManager

logger = logging.getLogger(__name__)

PipelineModule = Literal["setup", "discovery", "content"]


class PipelineOrchestrator:
    """Orchestrates execution of setup/discovery/content module runs."""

    def __init__(self, session: AsyncSession, project_id: str) -> None:
        self.session = session
        self.project_id = project_id
        self.task_manager = TaskManager()
        self._current_run: PipelineRun | None = None

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

        await self._assert_can_start_module(
            pipeline_module=run.pipeline_module,
            excluding_run_id=str(run.id),
        )

        module_cfg = self._module_config(run.pipeline_module)
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
            pipeline_module=run.pipeline_module,
            start_step=effective_start,
            end_step=effective_end,
            skip_steps=effective_skip,
            existing_config=run.steps_config,
            overrides=steps_config,
        )

        run.patch(
            self.session,
            PipelineRunPatchDTO.from_partial(
                {
                    "status": "running",
                    "started_at": datetime.now(timezone.utc),
                    "completed_at": None,
                    "error_message": None,
                    "paused_at_step": None,
                    "start_step": effective_start,
                    "end_step": effective_end,
                    "skip_steps": effective_skip,
                    "steps_config": effective_steps_config,
                }
            ),
        )

        await self.session.commit()
        self._current_run = run

        task_id = str(run.id)
        total_steps = len(
            [step for step in range(effective_start, effective_end + 1) if step not in effective_skip]
        )
        await self.task_manager.set_task_state(
            task_id=task_id,
            status="running",
            stage=f"{run.pipeline_module.capitalize()} pipeline started",
            project_id=self.project_id,
            pipeline_module=run.pipeline_module,
            source_topic_id=run.source_topic_id,
            current_step=effective_start,
            current_step_name=module_cfg["step_names"].get(effective_start),
            completed_steps=0,
            total_steps=total_steps,
            progress_percent=0.0,
            error_message=None,
        )

        try:
            if run.pipeline_module == "discovery":
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
                    pipeline_module=run.pipeline_module,
                    source_topic_id=run.source_topic_id,
                    current_step=effective_end,
                    current_step_name=module_cfg["step_names"].get(effective_end),
                    progress_percent=100.0,
                    error_message=None,
                )
                return run

            await self._run_step_range(
                run=run,
                module=run.pipeline_module,
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
                stage=f"{run.pipeline_module.capitalize()} pipeline completed",
                pipeline_module=run.pipeline_module,
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
                stage=f"{run.pipeline_module.capitalize()} pipeline paused",
                pipeline_module=run.pipeline_module,
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
        result = await self.session.execute(
            select(PipelineRun).where(
                PipelineRun.id == run_id,
                PipelineRun.project_id == self.project_id,
                PipelineRun.status == "paused",
            )
        )
        run = result.scalar_one_or_none()
        if run is None:
            raise ValueError("No paused pipeline found")
        if pipeline_module is not None and run.pipeline_module != pipeline_module:
            raise ValueError("Paused run module does not match requested module")

        if run.pipeline_module == "discovery":
            return await self.start_pipeline(
                run_id=run_id,
                pipeline_module=run.pipeline_module,
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

        module_cfg = self._module_config(run.pipeline_module)
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
            pipeline_module=run.pipeline_module,
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
        result = await self.session.execute(
            select(PipelineRun).where(
                PipelineRun.id == run_id,
                PipelineRun.project_id == self.project_id,
                PipelineRun.status == "running",
            )
        )
        run = result.scalar_one_or_none()
        if run is None:
            raise ValueError("No running pipeline found")

        run.patch(
            self.session,
            PipelineRunPatchDTO.from_partial({"status": "paused"}),
        )
        await self.session.commit()
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
            result = await self.session.execute(
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
        run = PipelineRun.create(
            self.session,
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
        await self.session.commit()
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
        supervisor = DiscoveryLoopSupervisor(
            session=self.session,
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

        existing_result = await self.session.execute(
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
            content_run = PipelineRun.create(
                self.session,
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
            await self.session.commit()
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
        result = await self.session.execute(
            select(StepExecution).where(
                StepExecution.pipeline_run_id == run_id,
                StepExecution.step_number == step_number,
                StepExecution.status == "completed",
            )
        )
        return result.scalar_one_or_none() is not None

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

    async def _update_run_status(self, run_id: Any, **updates: Any) -> None:
        async with get_session_context() as fresh_session:
            result = await fresh_session.execute(select(PipelineRun).where(PipelineRun.id == run_id))
            run = result.scalar_one()
            run.patch(fresh_session, PipelineRunPatchDTO.from_partial(updates))

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
        run = PipelineRun.create(
            self.session,
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
        await self.session.commit()
        self._current_run = run
        return run

    async def _get_last_completed_step(self, run_id: str) -> int | None:
        result = await self.session.execute(
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
        result = await self.session.execute(
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
        query = select(PipelineRun).where(
            PipelineRun.project_id == self.project_id,
            PipelineRun.pipeline_module == pipeline_module,
            PipelineRun.status == "running",
        )
        if excluding_run_id:
            query = query.where(PipelineRun.id != excluding_run_id)
        result = await self.session.execute(query)
        if result.scalar_one_or_none() is not None:
            raise PipelineAlreadyRunningError(self.project_id)

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
                "optional_steps": {8},
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
