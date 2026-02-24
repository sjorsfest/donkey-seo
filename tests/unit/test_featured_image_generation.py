"""Tests for featured image generation helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.featured_image_generation import (
    FeaturedImageGenerationService,
    modular_featured_image_payload,
    retry_with_backoff,
)


def test_locked_title_prefers_working_title() -> None:
    brief = SimpleNamespace(
        working_titles=["  Faster support operations in 2026  "],
        primary_keyword="support automation platform",
    )

    title = FeaturedImageGenerationService.locked_title_for_brief(brief)  # type: ignore[arg-type]

    assert title == "Faster support operations in 2026"


def test_locked_title_falls_back_to_keyword_titlecase() -> None:
    brief = SimpleNamespace(
        working_titles=[],
        primary_keyword="best helpdesk workflow automation tools",
    )

    title = FeaturedImageGenerationService.locked_title_for_brief(brief)  # type: ignore[arg-type]

    assert title == "Best Helpdesk Workflow Automation Tools"


def test_modular_featured_image_payload_adds_signed_url() -> None:
    featured_image = SimpleNamespace(
        object_key="projects/p1/content-images/b1/hash.png",
        mime_type="image/png",
        width=1200,
        height=630,
        byte_size=1234,
        sha256="abc",
        title_text="Title",
        template_version="1.0",
        source="llm_template_spec",
        style_variant_id="variant-1",
    )

    payload = modular_featured_image_payload(
        featured_image=featured_image,  # type: ignore[arg-type]
        signed_url="https://signed.example/image.png",
    )

    assert payload["object_key"].endswith(".png")
    assert payload["signed_url"].startswith("https://signed.example")


@pytest.mark.asyncio
async def test_retry_with_backoff_retries_until_success() -> None:
    attempts: list[int] = []

    async def _coro() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("boom")
        return "ok"

    result = await retry_with_backoff(
        attempts=3,
        backoff_ms=0,
        coro_factory=_coro,
    )

    assert result == "ok"
    assert len(attempts) == 3
