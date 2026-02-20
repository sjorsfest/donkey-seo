"""Unit tests for user-facing task step number mapping."""

from app.services.task_manager import TaskManager


def test_display_step_number_maps_discovery_to_one_based() -> None:
    assert TaskManager._display_step_number(pipeline_module="discovery", current_step=2) == 1
    assert TaskManager._display_step_number(pipeline_module="discovery", current_step=8) == 7


def test_display_step_number_maps_setup_to_one_based() -> None:
    assert TaskManager._display_step_number(pipeline_module="setup", current_step=0) == 1
    assert TaskManager._display_step_number(pipeline_module="setup", current_step=1) == 2


def test_display_step_number_keeps_content_unchanged() -> None:
    assert TaskManager._display_step_number(pipeline_module="content", current_step=1) == 1
    assert TaskManager._display_step_number(pipeline_module="content", current_step=3) == 3
