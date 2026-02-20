"""Unit tests for pipeline worker module selection."""

from app.workers.pipeline_worker import resolve_modules


def test_resolve_modules_defaults_to_all() -> None:
    assert resolve_modules(None) == ["setup", "discovery", "content"]


def test_resolve_modules_all_wins() -> None:
    assert resolve_modules(["discovery", "all"]) == ["setup", "discovery", "content"]


def test_resolve_modules_deduplicates_preserving_order() -> None:
    assert resolve_modules(["content", "setup", "content"]) == ["content", "setup"]

