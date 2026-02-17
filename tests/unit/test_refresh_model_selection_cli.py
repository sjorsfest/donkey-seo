"""Tests for refresh_model_selection CLI helpers."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from app.config import settings
from app.services.model_selector.types import SelectionSnapshot
from scripts import refresh_model_selection as refresh


def test_get_environment_max_price_uses_environment_specific_caps() -> None:
    """Max-price helper returns per-environment values."""
    original_dev = settings.model_selector_max_price_dev
    original_staging = settings.model_selector_max_price_staging
    original_prod = settings.model_selector_max_price_prod

    try:
        settings.model_selector_max_price_dev = 0.0
        settings.model_selector_max_price_staging = 0.25
        settings.model_selector_max_price_prod = 1.5

        assert refresh.get_environment_max_price("development") == 0.0
        assert refresh.get_environment_max_price("staging") == 0.25
        assert refresh.get_environment_max_price("production") == 1.5
    finally:
        settings.model_selector_max_price_dev = original_dev
        settings.model_selector_max_price_staging = original_staging
        settings.model_selector_max_price_prod = original_prod


def test_write_snapshot_atomically_keeps_original_when_replace_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Atomic write does not corrupt existing snapshot if replace fails."""
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text('{"version": "old"}', encoding="utf-8")

    snapshot = SelectionSnapshot(
        version="1",
        generated_at="2026-02-16T00:00:00+00:00",
        environments={},
    )

    def raising_replace(self: Path, target: Path) -> Path:
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", raising_replace)

    try:
        refresh.write_snapshot_atomically(snapshot_path, snapshot)
    except OSError:
        pass

    assert snapshot_path.read_text(encoding="utf-8") == '{"version": "old"}'


def test_async_main_dry_run_does_not_write_snapshot(tmp_path: Path, monkeypatch) -> None:
    """Dry-run mode computes summary without writing output file."""
    usecases_path = tmp_path / "agent_usecases.yaml"
    usecases_path.write_text("agents: {}\n", encoding="utf-8")

    snapshot_path = tmp_path / "model_selection_snapshot.json"

    async def fake_build_snapshot(**kwargs):
        snapshot = SelectionSnapshot(
            version="1",
            generated_at="2026-02-16T00:00:00+00:00",
            environments={},
        )
        summary = {
            "environments": 1,
            "agents": 1,
            "fallbacks": 0,
            "arena_degraded": 0,
            "openrouter_failures": 0,
        }
        return snapshot, summary

    monkeypatch.setattr(refresh, "build_snapshot", fake_build_snapshot)

    original_snapshot_path = settings.model_selector_snapshot_path
    original_argv = sys.argv[:]

    try:
        settings.model_selector_snapshot_path = str(snapshot_path)
        sys.argv = [
            "refresh_model_selection.py",
            "--dry-run",
            "--usecases-path",
            str(usecases_path),
            "--env",
            "development",
        ]

        rc = asyncio.run(refresh.async_main())
        assert rc == 0
        assert snapshot_path.exists() is False
    finally:
        settings.model_selector_snapshot_path = original_snapshot_path
        sys.argv = original_argv
