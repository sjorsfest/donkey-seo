"""Tests for author profile image R2 storage integration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.integrations.author_image_store import AuthorImageStore


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        cloudflare_r2_account_id="acct",
        cloudflare_r2_access_key_id="key",
        cloudflare_r2_secret_access_key="secret",
        cloudflare_r2_bucket="private-bucket",
        cloudflare_r2_region="auto",
        brand_asset_max_bytes=5_000_000,
        signed_url_ttl_seconds=120,
    )


@pytest.mark.asyncio
async def test_ingest_source_url_uploads_profile_image(monkeypatch: pytest.MonkeyPatch) -> None:
    store = AuthorImageStore(app_settings=_settings())

    async def _fake_download(_source_url: str) -> tuple[bytes, str]:
        return b"author-image", "image/png"

    class _FakeClient:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def put_object(self, **kwargs: dict) -> None:
            self.calls.append(kwargs)

    fake_client = _FakeClient()
    store._client = fake_client
    monkeypatch.setattr(store, "_download_image_bytes", _fake_download)

    metadata = await store.ingest_source_url(
        project_id="project_1",
        author_id="author_1",
        source_url="https://example.com/avatar.png",
    )

    assert metadata["profile_image_object_key"].startswith("projects/project_1/authors/author_1/")
    assert metadata["profile_image_mime_type"] == "image/png"
    assert len(fake_client.calls) == 1


def test_create_signed_read_url_uses_s3_presign() -> None:
    store = AuthorImageStore(app_settings=_settings())

    class _FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict, int]] = []

        def generate_presigned_url(self, operation: str, Params: dict, ExpiresIn: int) -> str:  # noqa: N803
            self.calls.append((operation, Params, ExpiresIn))
            return "https://signed.example/author"

    fake_client = _FakeClient()
    store._client = fake_client

    signed_url = store.create_signed_read_url(
        object_key="projects/p1/authors/a1/avatar.png"
    )

    assert signed_url == "https://signed.example/author"
    assert fake_client.calls == [
        (
            "get_object",
            {"Bucket": "private-bucket", "Key": "projects/p1/authors/a1/avatar.png"},
            120,
        )
    ]
