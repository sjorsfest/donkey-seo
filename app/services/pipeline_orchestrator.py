"""Pipeline orchestrator for coordinating step execution."""

from datetime import datetime, timezone
from typing import Any
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    PipelineAlreadyRunningError,
    StepNotFoundError,
    StepPreconditionError,
)
from app.models.pipeline import PipelineRun, StepExecution
from app.models.project import Project

# Step service imports
from app.services.steps.step_00_setup import Step00SetupService, SetupInput
from app.services.steps.step_01_brand import Step01BrandService, BrandInput
from app.services.steps.step_02_seeds import Step02SeedsService, SeedsInput
from app.services.steps.step_03_expansion import Step03ExpansionService, ExpansionInput
from app.services.steps.step_04_metrics import Step04MetricsService, MetricsInput
from app.services.steps.step_05_intent import Step05IntentService, IntentInput
from app.services.steps.step_06_clustering import Step06ClusteringService, ClusteringInput
from app.services.steps.step_07_prioritization import Step07PrioritizationService, PrioritizationInput
from app.services.steps.step_12_brief import Step12BriefService, BriefInput
from app.services.steps.step_13_templates import Step13TemplatesService, TemplatesInput


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
    14: "gsc_integration",
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
    14: [0],
}

# Optional steps (can be skipped)
OPTIONAL_STEPS = {8, 9, 10, 11, 14}

# Steps requiring OAuth
OAUTH_STEPS = {14}


class PipelineOrchestrator:
    """Orchestrates the execution of pipeline steps."""

    def __init__(self, session: AsyncSession, project_id: str) -> None:
        self.session = session
        self.project_id = project_id
        self._current_run: PipelineRun | None = None

    async def start_pipeline(
        self,
        start_step: int = 0,
        end_step: int | None = None,
        skip_steps: list[int] | None = None,
    ) -> PipelineRun:
        """Start a new pipeline run."""
        end_step = end_step or 13
        skip_steps = skip_steps or []

        # Check for running pipeline
        result = await self.session.execute(
            select(PipelineRun).where(
                PipelineRun.project_id == self.project_id,
                PipelineRun.status == "running",
            )
        )
        if result.scalar_one_or_none():
            raise PipelineAlreadyRunningError(self.project_id)

        # Create pipeline run
        run = PipelineRun(
            project_id=uuid.UUID(self.project_id),
            status="running",
            started_at=datetime.now(timezone.utc),
            start_step=start_step,
            end_step=end_step,
            skip_steps=skip_steps,
            steps_config={
                "start": start_step,
                "end": end_step,
                "skip": skip_steps,
            },
        )
        self.session.add(run)
        await self.session.flush()

        self._current_run = run

        # Execute steps
        try:
            for step_num in range(start_step, end_step + 1):
                if await self._should_skip_step(step_num, skip_steps):
                    await self._create_skipped_execution(run, step_num)
                    continue

                await self._execute_step(run, step_num)

            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc)

        except Exception as e:
            run.status = "failed"
            run.completed_at = datetime.now(timezone.utc)
            raise

        finally:
            await self.session.commit()

        return run

    async def run_single_step(self, step_number: int) -> StepExecution:
        """Run a single step independently."""
        if step_number not in STEP_NAMES:
            raise StepNotFoundError(step_number)

        # Verify dependencies
        await self._verify_dependencies(step_number)

        # Get or create pipeline run
        run = await self._get_or_create_run()

        return await self._execute_step(run, step_number)

    async def resume_pipeline(self, run_id: str) -> PipelineRun:
        """Resume a paused pipeline."""
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

        run.status = "running"
        self._current_run = run

        # Find last completed step and resume from next
        last_completed = await self._get_last_completed_step(run)
        start_step = last_completed + 1 if last_completed is not None else 0
        end_step = run.end_step or 13
        skip_steps = run.skip_steps or []

        try:
            for step_num in range(start_step, end_step + 1):
                if await self._should_skip_step(step_num, skip_steps):
                    continue
                await self._execute_step(run, step_num)

            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc)

        except Exception:
            run.status = "failed"
            run.completed_at = datetime.now(timezone.utc)
            raise

        finally:
            await self.session.commit()

        return run

    async def pause_pipeline(self) -> None:
        """Pause the current pipeline."""
        if self._current_run:
            self._current_run.status = "paused"
            await self.session.commit()

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
            if step_number == 14:
                # Skip GSC if no OAuth tokens
                return not await self._has_oauth_tokens()

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
        """Execute a single step with proper service instantiation."""
        execution = StepExecution(
            pipeline_run_id=run.id,
            step_number=step_number,
            step_name=STEP_NAMES[step_number],
            status="pending",
        )
        self.session.add(execution)
        await self.session.flush()

        # Import and instantiate the appropriate step service
        service = await self._get_step_service(step_number, execution)

        if service is None:
            # Step not implemented yet - mark as skipped
            execution.status = "skipped"
            execution.progress_message = "Step not implemented"
            await self.session.commit()
            return execution

        # Get input data for the step
        input_data = await self._get_step_input(step_number)

        # Run the step
        await service.run(input_data)

        return execution

    async def _get_step_service(
        self,
        step_number: int,
        execution: StepExecution,
    ) -> Any | None:
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
            # Steps 8-11 are optional and not yet implemented
            # 8: Step08SerpService,
            # 9: Step09InventoryService,
            # 10: Step10CannibalizationService,
            # 11: Step11LinkingService,
            12: Step12BriefService,
            13: Step13TemplatesService,
            # 14: Step14GscService,  # Requires OAuth
        }

        service_class = step_services.get(step_number)
        if service_class is None:
            return None

        return service_class(
            session=self.session,
            project_id=self.project_id,
            execution=execution,
        )

    async def _get_step_input(self, step_number: int) -> Any:
        """Get input data for a step from previous step outputs."""
        # Load project
        result = await self.session.execute(
            select(Project).where(Project.id == self.project_id)
        )
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
            # Steps 8-11 would have their own input types
            12: BriefInput(project_id=self.project_id),
            13: TemplatesInput(project_id=self.project_id),
        }

        return step_inputs.get(step_number, {"project_id": self.project_id})

    async def _create_skipped_execution(
        self,
        run: PipelineRun,
        step_number: int,
        reason: str = "Skipped by configuration",
    ) -> StepExecution:
        """Create a skipped step execution record."""
        execution = StepExecution(
            pipeline_run_id=run.id,
            step_number=step_number,
            step_name=STEP_NAMES[step_number],
            status="skipped",
            progress_message=reason,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )
        self.session.add(execution)
        await self.session.flush()
        return execution

    async def _get_or_create_run(self) -> PipelineRun:
        """Get existing run or create a new one for single step execution."""
        if self._current_run:
            return self._current_run

        run = PipelineRun(
            project_id=uuid.UUID(self.project_id),
            status="running",
            started_at=datetime.now(timezone.utc),
        )
        self.session.add(run)
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

    async def _has_existing_content(self) -> bool:
        """Check if the project has existing content to inventory."""
        # TODO: Implement actual check
        return False

    async def _has_oauth_tokens(self) -> bool:
        """Check if OAuth tokens are configured."""
        # TODO: Implement actual check
        return False
