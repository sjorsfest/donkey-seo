"""Unit tests for user-facing task step number mapping."""

from app.services.task_manager import TaskManager


def test_display_step_number_keeps_discovery_unchanged() -> None:
    assert TaskManager._display_step_number(pipeline_module="discovery", current_step=1) == 1
    assert TaskManager._display_step_number(pipeline_module="discovery", current_step=7) == 7


def test_display_step_number_keeps_setup_unchanged() -> None:
    assert TaskManager._display_step_number(pipeline_module="setup", current_step=1) == 1
    assert TaskManager._display_step_number(pipeline_module="setup", current_step=5) == 5


def test_display_step_number_keeps_content_unchanged() -> None:
    assert TaskManager._display_step_number(pipeline_module="content", current_step=1) == 1
    assert TaskManager._display_step_number(pipeline_module="content", current_step=3) == 3
