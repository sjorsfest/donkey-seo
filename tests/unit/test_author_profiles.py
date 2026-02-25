"""Unit tests for author profile helper utilities."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.author_profiles import (
    author_modular_document_payload,
    choose_random_author,
    enrich_modular_document_with_signed_author_image,
)


def test_choose_random_author_returns_none_for_empty_sequence() -> None:
    assert choose_random_author([]) is None


def test_author_modular_payload_includes_profile_image_when_available() -> None:
    author = SimpleNamespace(
        id="author_1",
        name="Jamie Doe",
        bio="SEO strategist",
        social_urls={"linkedin": "https://linkedin.com/in/jamie"},
        basic_info={"title": "Head of Content"},
        profile_image_object_key="projects/p1/authors/a1/avatar.png",
        profile_image_mime_type="image/png",
        profile_image_width=400,
        profile_image_height=400,
        profile_image_byte_size=12000,
        profile_image_sha256="abc",
    )

    payload = author_modular_document_payload(author)  # type: ignore[arg-type]

    assert payload["id"] == "author_1"
    assert payload["name"] == "Jamie Doe"
    assert payload["profile_image"]["object_key"] == "projects/p1/authors/a1/avatar.png"


def test_enrich_modular_document_adds_signed_author_profile_image_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeStore:
        def create_signed_read_url(self, *, object_key: str) -> str:
            assert object_key == "projects/p1/authors/a1/avatar.png"
            return "https://signed.example/avatar"

    monkeypatch.setattr(
        "app.services.author_profiles.AuthorImageStore",
        lambda: _FakeStore(),
    )

    enriched = enrich_modular_document_with_signed_author_image(
        {
            "schema_version": "1.0",
            "author": {
                "id": "author_1",
                "name": "Jamie Doe",
                "profile_image": {"object_key": "projects/p1/authors/a1/avatar.png"},
            },
        }
    )

    author_payload = enriched.get("author")
    assert isinstance(author_payload, dict)
    profile_image = author_payload.get("profile_image")
    assert isinstance(profile_image, dict)
    assert profile_image["signed_url"] == "https://signed.example/avatar"
