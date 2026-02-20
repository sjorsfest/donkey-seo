"""Private Cloudflare R2 asset ingestion and signed URL utilities."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import httpx

from app.config import Settings, settings
from app.core.ids import generate_cuid

logger = logging.getLogger(__name__)


class BrandAssetStoreError(RuntimeError):
    """Base error for brand asset storage operations."""


class BrandAssetStoreConfigError(BrandAssetStoreError):
    """Raised when required R2 settings are missing."""


class BrandAssetStore:
    """Ingest image assets into private R2 storage and mint signed URLs."""

    _MIME_TO_EXTENSION = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
        "image/svg+xml": ".svg",
        "image/gif": ".gif",
        "image/x-icon": ".ico",
        "image/vnd.microsoft.icon": ".ico",
        "image/avif": ".avif",
    }

    def __init__(self, app_settings: Settings | None = None) -> None:
        self.settings = app_settings or settings
        self._client: Any | None = None

    async def ingest_asset_candidates(
        self,
        *,
        project_id: str,
        asset_candidates: list[dict[str, Any]],
        existing_assets: list[dict[str, Any]] | None = None,
        origin: str = "step_01_auto",
    ) -> list[dict[str, Any]]:
        """Ingest discovered candidates into private R2 and return deduplicated metadata."""
        self._validate_config()

        assets_by_sha: dict[str, dict[str, Any]] = {}
        if existing_assets:
            for asset in existing_assets:
                sha = str(asset.get("sha256") or "").strip()
                if not sha:
                    continue
                assets_by_sha[sha] = dict(asset)

        sorted_candidates = sorted(
            asset_candidates,
            key=lambda item: float(item.get("role_confidence") or 0.0),
            reverse=True,
        )

        for candidate in sorted_candidates:
            if len(assets_by_sha) >= self.settings.brand_assets_max_count:
                break

            source_url = str(candidate.get("url") or "").strip()
            if not source_url:
                continue

            try:
                payload, mime_type = await self._download_image_bytes(source_url)
            except Exception as exc:
                logger.warning(
                    "Skipping asset candidate due to download failure",
                    extra={"source_url": source_url, "error": str(exc)},
                )
                continue

            sha256 = hashlib.sha256(payload).hexdigest()
            if sha256 in assets_by_sha:
                assets_by_sha[sha256] = self._merge_existing_asset(
                    assets_by_sha[sha256],
                    role=str(candidate.get("role") or "reference"),
                    role_confidence=float(candidate.get("role_confidence") or 0.0),
                    source_url=source_url,
                )
                continue

            extension = self._resolve_extension(mime_type, source_url)
            object_key = f"projects/{project_id}/brand-assets/{sha256}{extension}"
            width, height = self._detect_dimensions(payload)

            await asyncio.to_thread(
                self._upload_object,
                object_key,
                payload,
                mime_type,
                sha256,
            )

            now_iso = datetime.now(timezone.utc).isoformat()
            assets_by_sha[sha256] = {
                "asset_id": generate_cuid(),
                "object_key": object_key,
                "sha256": sha256,
                "mime_type": mime_type,
                "byte_size": len(payload),
                "width": width,
                "height": height,
                "role": str(candidate.get("role") or "reference"),
                "role_confidence": float(candidate.get("role_confidence") or 0.5),
                "source_url": source_url,
                "origin": str(candidate.get("origin") or origin),
                "ingested_at": now_iso,
            }
            logger.info(
                "Brand asset ingested",
                extra={"asset_id": assets_by_sha[sha256]["asset_id"], "object_key": object_key},
            )

        merged = list(assets_by_sha.values())
        merged.sort(
            key=lambda item: float(item.get("role_confidence") or 0.0),
            reverse=True,
        )
        return merged[: self.settings.brand_assets_max_count]

    async def ingest_source_urls(
        self,
        *,
        project_id: str,
        source_urls: list[str],
        existing_assets: list[dict[str, Any]] | None = None,
        role: str = "reference",
        origin: str = "manual_url",
    ) -> list[dict[str, Any]]:
        """Ingest manually supplied source URLs via server-side fetch."""
        candidates = [
            {
                "url": source_url,
                "role": role,
                "role_confidence": 0.6,
                "origin": origin,
            }
            for source_url in source_urls
            if str(source_url).strip()
        ]
        return await self.ingest_asset_candidates(
            project_id=project_id,
            asset_candidates=candidates,
            existing_assets=existing_assets,
            origin=origin,
        )

    def create_signed_read_url(self, *, object_key: str, ttl_seconds: int | None = None) -> str:
        """Mint short-lived signed GET URL for a private object."""
        self._validate_config()
        expires_in = int(ttl_seconds or self.settings.signed_url_ttl_seconds)
        client = self._get_client()
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.settings.cloudflare_r2_bucket, "Key": object_key},
            ExpiresIn=expires_in,
        )

    def create_signed_upload_url(
        self,
        *,
        object_key: str,
        ttl_seconds: int | None = None,
        content_type: str | None = None,
    ) -> str:
        """Mint signed PUT URL for future upload support."""
        self._validate_config()
        expires_in = int(ttl_seconds or self.settings.signed_url_ttl_seconds)
        params: dict[str, Any] = {
            "Bucket": self.settings.cloudflare_r2_bucket,
            "Key": object_key,
        }
        if content_type:
            params["ContentType"] = content_type

        client = self._get_client()
        return client.generate_presigned_url(
            "put_object",
            Params=params,
            ExpiresIn=expires_in,
        )

    async def _download_image_bytes(self, source_url: str) -> tuple[bytes, str]:
        timeout = httpx.Timeout(30.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(source_url)
            response.raise_for_status()

        content_length = response.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > self.settings.brand_asset_max_bytes:
                    raise BrandAssetStoreError("asset exceeds maximum byte size")
            except ValueError:
                logger.debug("Ignoring non-integer content-length header")

        content = response.content
        if len(content) > self.settings.brand_asset_max_bytes:
            raise BrandAssetStoreError("asset exceeds maximum byte size")

        mime_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if not mime_type.startswith("image/"):
            raise BrandAssetStoreError(f"unsupported content type: {mime_type or 'unknown'}")

        return content, mime_type

    @staticmethod
    def _merge_existing_asset(
        existing_asset: dict[str, Any],
        *,
        role: str,
        role_confidence: float,
        source_url: str,
    ) -> dict[str, Any]:
        merged = dict(existing_asset)
        existing_confidence = float(merged.get("role_confidence") or 0.0)
        if role_confidence > existing_confidence:
            merged["role"] = role
            merged["role_confidence"] = role_confidence
        if not merged.get("source_url"):
            merged["source_url"] = source_url
        return merged

    def _upload_object(self, object_key: str, payload: bytes, mime_type: str, sha256: str) -> None:
        client = self._get_client()
        client.put_object(
            Bucket=self.settings.cloudflare_r2_bucket,
            Key=object_key,
            Body=payload,
            ContentType=mime_type,
            Metadata={"sha256": sha256},
        )

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:
        try:
            import boto3
            from botocore.config import Config
        except Exception as exc:
            raise BrandAssetStoreConfigError(
                "boto3 is required for brand asset storage operations"
            ) from exc

        endpoint_url = f"https://{self.settings.cloudflare_r2_account_id}.r2.cloudflarestorage.com"
        return boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=self.settings.cloudflare_r2_region,
            aws_access_key_id=self.settings.cloudflare_r2_access_key_id,
            aws_secret_access_key=self.settings.cloudflare_r2_secret_access_key,
            config=Config(signature_version="s3v4"),
        )

    @classmethod
    def _resolve_extension(cls, mime_type: str, source_url: str) -> str:
        if mime_type in cls._MIME_TO_EXTENSION:
            return cls._MIME_TO_EXTENSION[mime_type]

        source_url_lower = source_url.lower()
        for extension in (".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif", ".ico", ".avif"):
            if source_url_lower.endswith(extension):
                return ".jpg" if extension == ".jpeg" else extension
        return ".bin"

    @staticmethod
    def _detect_dimensions(payload: bytes) -> tuple[int | None, int | None]:
        try:
            from PIL import Image

            with Image.open(BytesIO(payload)) as image:
                width, height = image.size
                return int(width), int(height)
        except Exception:
            return None, None

    def _validate_config(self) -> None:
        required = {
            "cloudflare_r2_account_id": self.settings.cloudflare_r2_account_id,
            "cloudflare_r2_access_key_id": self.settings.cloudflare_r2_access_key_id,
            "cloudflare_r2_secret_access_key": self.settings.cloudflare_r2_secret_access_key,
            "cloudflare_r2_bucket": self.settings.cloudflare_r2_bucket,
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise BrandAssetStoreConfigError(
                "Missing required R2 config: " + ", ".join(sorted(missing))
            )
