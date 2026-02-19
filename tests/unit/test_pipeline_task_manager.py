"""Unit tests for module-specific pipeline task managers."""

from app.services.pipeline_task_manager import (
    get_content_pipeline_task_manager,
    get_discovery_pipeline_task_manager,
)


def test_discovery_task_manager_singleton_and_module() -> None:
    manager_a = get_discovery_pipeline_task_manager()
    manager_b = get_discovery_pipeline_task_manager()

    assert manager_a is manager_b
    assert manager_a.pipeline_module == "discovery"


def test_content_task_manager_singleton_and_module() -> None:
    manager_a = get_content_pipeline_task_manager()
    manager_b = get_content_pipeline_task_manager()

    assert manager_a is manager_b
    assert manager_a.pipeline_module == "content"
