"""Tests for private R2 asset storage integration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.integrations.asset_store import BrandAssetStore


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        cloudflare_r2_account_id="acct",
        cloudflare_r2_access_key_id="key",
        cloudflare_r2_secret_access_key="secret",
        cloudflare_r2_bucket="private-bucket",
        cloudflare_r2_region="auto",
        brand_asset_max_bytes=5_000_000,
        brand_assets_max_count=8,
        signed_url_ttl_seconds=90,
    )


@pytest.mark.asyncio
async def test_ingest_asset_candidates_deduplicates_by_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    store = BrandAssetStore(app_settings=_settings())

    async def _fake_download(_source_url: str) -> tuple[bytes, str]:
        return b"same-image-payload", "image/png"

    uploaded_keys: list[str] = []

    def _fake_upload(object_key: str, _payload: bytes, _mime: str, _sha: str) -> None:
        uploaded_keys.append(object_key)

    monkeypatch.setattr(store, "_download_image_bytes", _fake_download)
    monkeypatch.setattr(store, "_upload_object", _fake_upload)

    assets = await store.ingest_asset_candidates(
        project_id="project-1",
        asset_candidates=[
            {"url": "https://example.com/logo-a.png", "role": "logo", "role_confidence": 0.9},
            {"url": "https://example.com/logo-b.png", "role": "logo", "role_confidence": 0.8},
        ],
    )

    assert len(assets) == 1
    assert len(uploaded_keys) == 1
    assert assets[0]["object_key"].startswith("projects/project-1/brand-assets/")


def test_create_signed_read_url_uses_s3_presign() -> None:
    store = BrandAssetStore(app_settings=_settings())

    class _FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict, int]] = []

        def generate_presigned_url(self, operation: str, Params: dict, ExpiresIn: int) -> str:  # noqa: N803
            self.calls.append((operation, Params, ExpiresIn))
            return "https://signed.example/read"

    fake_client = _FakeClient()
    store._client = fake_client

    signed_url = store.create_signed_read_url(object_key="projects/p1/brand-assets/abc.png")

    assert signed_url == "https://signed.example/read"
    assert fake_client.calls == [
        (
            "get_object",
            {"Bucket": "private-bucket", "Key": "projects/p1/brand-assets/abc.png"},
            90,
        )
    ]
