"""Unit tests for pipeline response step-number display mapping."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from app.api.v1.pipeline.routes import _pipeline_run_response


def test_pipeline_run_response_maps_discovery_steps_to_one_based() -> None:
    now = datetime.now(timezone.utc)
    run = SimpleNamespace(
        id="run-1",
        project_id="project-1",
        pipeline_module="discovery",
        parent_run_id=None,
        source_topic_id=None,
        status="running",
        started_at=now,
        completed_at=None,
        start_step=1,
        end_step=7,
        skip_steps=[7],
        step_executions=[
            SimpleNamespace(
                id="exec-1",
                step_number=1,
                step_name="seed_topics",
                status="running",
                progress_percent=30.0,
                progress_message="Generating seed keywords...",
                items_processed=0,
                items_total=None,
                started_at=now,
                completed_at=None,
                error_message=None,
            )
        ],
        created_at=now,
    )

    response = _pipeline_run_response(run)

    assert response.start_step == 1
    assert response.end_step == 7
    assert response.skip_steps == [7]
    assert response.step_executions[0].step_number == 1
