"""Unit tests for module-specific pipeline task managers."""

from app.services.pipeline_task_manager import (
    PipelineTaskJob,
    get_content_pipeline_task_manager,
    get_discovery_pipeline_task_manager,
    get_pipeline_task_manager,
    get_setup_pipeline_task_manager,
)


def test_setup_task_manager_singleton_and_module() -> None:
    manager_a = get_setup_pipeline_task_manager()
    manager_b = get_setup_pipeline_task_manager()

    assert manager_a is manager_b
    assert manager_a.pipeline_module == "setup"


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


def test_get_pipeline_task_manager_dispatch() -> None:
    assert get_pipeline_task_manager("setup") is get_setup_pipeline_task_manager()
    assert get_pipeline_task_manager("discovery") is get_discovery_pipeline_task_manager()
    assert get_pipeline_task_manager("content") is get_content_pipeline_task_manager()


def test_pipeline_task_job_serialize_roundtrip() -> None:
    manager = get_setup_pipeline_task_manager()
    payload = manager._serialize_job(
        PipelineTaskJob(kind="start", project_id="proj_1", run_id="run_1")
    )
    decoded = manager._deserialize_job(payload)

    assert decoded.kind == "start"
    assert decoded.project_id == "proj_1"
    assert decoded.run_id == "run_1"
