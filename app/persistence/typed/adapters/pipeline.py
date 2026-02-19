"""Pipeline model write adapters."""

from __future__ import annotations

from app.models.generated_dtos import (
    DiscoveryIterationLearningCreateDTO,
    DiscoveryIterationLearningPatchDTO,
    DiscoveryTopicSnapshotCreateDTO,
    DiscoveryTopicSnapshotPatchDTO,
    PipelineRunCreateDTO,
    PipelineRunPatchDTO,
    StepExecutionCreateDTO,
    StepExecutionPatchDTO,
)
from app.models.discovery_learning import DiscoveryIterationLearning
from app.models.discovery_snapshot import DiscoveryTopicSnapshot
from app.models.pipeline import PipelineRun, StepExecution
from app.persistence.typed.adapters._base import BaseWriteAdapter

PIPELINE_RUN_PATCH_ALLOWLIST = {
    "pipeline_module",
    "parent_run_id",
    "source_topic_id",
    "status",
    "error_message",
    "paused_at_step",
    "steps_config",
    "start_step",
    "end_step",
    "skip_steps",
    "started_at",
    "completed_at",
}

STEP_EXECUTION_PATCH_ALLOWLIST = {
    "step_name",
    "started_at",
    "completed_at",
    "status",
    "progress_percent",
    "progress_message",
    "items_processed",
    "items_total",
    "result_summary",
    "error_message",
    "error_traceback",
    "checkpoint_data",
}

DISCOVERY_TOPIC_SNAPSHOT_PATCH_ALLOWLIST = {
    "iteration_index",
    "source_topic_id",
    "topic_name",
    "fit_tier",
    "fit_score",
    "keyword_difficulty",
    "domain_diversity",
    "validated_intent",
    "validated_page_type",
    "top_domains",
    "decision",
    "rejection_reasons",
}

DISCOVERY_ITERATION_LEARNING_PATCH_ALLOWLIST = {
    "iteration_index",
    "source_capability",
    "source_agent",
    "learning_key",
    "learning_type",
    "polarity",
    "status",
    "title",
    "detail",
    "recommendation",
    "confidence",
    "novelty_score",
    "baseline_metric",
    "current_metric",
    "delta_metric",
    "applies_to_capabilities",
    "applies_to_agents",
    "evidence",
}

_PIPELINE_RUN_ADAPTER = BaseWriteAdapter[
    PipelineRun,
    PipelineRunCreateDTO,
    PipelineRunPatchDTO,
](
    model_cls=PipelineRun,
    patch_allowlist=PIPELINE_RUN_PATCH_ALLOWLIST,
)

_STEP_EXECUTION_ADAPTER = BaseWriteAdapter[
    StepExecution,
    StepExecutionCreateDTO,
    StepExecutionPatchDTO,
](
    model_cls=StepExecution,
    patch_allowlist=STEP_EXECUTION_PATCH_ALLOWLIST,
)

_DISCOVERY_TOPIC_SNAPSHOT_ADAPTER = BaseWriteAdapter[
    DiscoveryTopicSnapshot,
    DiscoveryTopicSnapshotCreateDTO,
    DiscoveryTopicSnapshotPatchDTO,
](
    model_cls=DiscoveryTopicSnapshot,
    patch_allowlist=DISCOVERY_TOPIC_SNAPSHOT_PATCH_ALLOWLIST,
)

_DISCOVERY_ITERATION_LEARNING_ADAPTER = BaseWriteAdapter[
    DiscoveryIterationLearning,
    DiscoveryIterationLearningCreateDTO,
    DiscoveryIterationLearningPatchDTO,
](
    model_cls=DiscoveryIterationLearning,
    patch_allowlist=DISCOVERY_ITERATION_LEARNING_PATCH_ALLOWLIST,
)


def register() -> None:
    """Register pipeline adapters."""
    from app.persistence.typed.registry import register_adapter

    register_adapter(PipelineRun, _PIPELINE_RUN_ADAPTER)
    register_adapter(StepExecution, _STEP_EXECUTION_ADAPTER)
    register_adapter(DiscoveryTopicSnapshot, _DISCOVERY_TOPIC_SNAPSHOT_ADAPTER)
    register_adapter(DiscoveryIterationLearning, _DISCOVERY_ITERATION_LEARNING_ADAPTER)
