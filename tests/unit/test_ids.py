"""Unit tests for CUID generation utilities."""

from __future__ import annotations

from app.core.ids import generate_cuid


def test_generate_cuid_format_and_uniqueness() -> None:
    ids = [generate_cuid() for _ in range(200)]

    assert len(ids) == len(set(ids))
    assert all(len(item) == 24 for item in ids)
    assert all(item.startswith("c") for item in ids)
    assert all(item.isalnum() and item == item.lower() for item in ids)
