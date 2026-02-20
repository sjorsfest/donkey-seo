"""Unit tests for task progress estimation in BaseStepService."""

from app.services.steps.base_step import BaseStepService


def test_estimate_run_progress_percent_returns_none_without_task_payload() -> None:
    value = BaseStepService._estimate_run_progress_percent(
        task_status=None,
        step_percent=35.0,
    )

    assert value is None


def test_estimate_run_progress_percent_uses_completed_plus_current_step_fraction() -> None:
    value = BaseStepService._estimate_run_progress_percent(
        task_status={"completed_steps": 1, "total_steps": 2},
        step_percent=40.0,
    )

    # (1 + 0.4) / 2 = 70%
    assert value == 70.0


def test_estimate_run_progress_percent_clamps_overflow_to_100() -> None:
    value = BaseStepService._estimate_run_progress_percent(
        task_status={"completed_steps": 3, "total_steps": 3},
        step_percent=90.0,
    )

    assert value == 100.0
