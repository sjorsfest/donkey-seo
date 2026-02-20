"""Unit tests for queue health endpoint."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.config import settings
from app.main import create_app


class _FakeManager:
    def __init__(self, *, queued: int, limit: int, workers: int) -> None:
        self._queued = queued
        self.queue_size_limit = limit
        self.worker_count = workers

    async def get_queue_size(self) -> int:
        return self._queued


def test_health_queue_reports_module_queue_lengths(monkeypatch: Any) -> None:
    """Queue health endpoint returns per-module and total queue stats."""
    original_environment = settings.environment
    settings.environment = "production"
    try:
        monkeypatch.setattr(
            "app.main.get_setup_pipeline_task_manager",
            lambda: _FakeManager(queued=2, limit=50, workers=1),
        )
        monkeypatch.setattr(
            "app.main.get_discovery_pipeline_task_manager",
            lambda: _FakeManager(queued=3, limit=100, workers=1),
        )
        monkeypatch.setattr(
            "app.main.get_content_pipeline_task_manager",
            lambda: _FakeManager(queued=5, limit=200, workers=1),
        )

        with TestClient(create_app()) as client:
            response = client.get("/health/queue")
    finally:
        settings.environment = original_environment

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["total_queued"] == 10
    assert payload["queues"]["setup"] == {"queued": 2, "limit": 50, "workers": 1}
    assert payload["queues"]["discovery"] == {"queued": 3, "limit": 100, "workers": 1}
    assert payload["queues"]["content"] == {"queued": 5, "limit": 200, "workers": 1}

