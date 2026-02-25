"""Storage helpers for author profile images in private R2 storage."""

from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Any

import httpx

from app.config import Settings, settings


class AuthorImageStoreError(RuntimeError):
    """Base error for author image storage operations."""


class AuthorImageStoreConfigError(AuthorImageStoreError):
    """Raised when required cloud storage settings are missing."""


class AuthorImageStore:
    """Upload author profile images and mint short-lived signed read URLs."""

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

    async def ingest_source_url(
        self,
        *,
        project_id: str,
        author_id: str,
        source_url: str,
    ) -> dict[str, Any]:
        """Download a profile image from URL, upload to R2, and return immutable metadata."""
        self._validate_config()

        payload, mime_type = await self._download_image_bytes(source_url)
        sha256 = hashlib.sha256(payload).hexdigest()
        extension = self._resolve_extension(mime_type, source_url)
        object_key = f"projects/{project_id}/authors/{author_id}/{sha256}{extension}"
        width, height = self._detect_dimensions(payload)

        client = self._get_client()
        client.put_object(
            Bucket=self.settings.cloudflare_r2_bucket,
            Key=object_key,
            Body=payload,
            ContentType=mime_type,
            Metadata={"sha256": sha256, "source_url": source_url},
        )

        return {
            "profile_image_source_url": source_url,
            "profile_image_object_key": object_key,
            "profile_image_mime_type": mime_type,
            "profile_image_width": width,
            "profile_image_height": height,
            "profile_image_byte_size": len(payload),
            "profile_image_sha256": sha256,
        }

    def create_signed_read_url(self, *, object_key: str, ttl_seconds: int | None = None) -> str:
        """Mint short-lived signed URL for GET access to a profile image object."""
        self._validate_config()
        expires_in = int(ttl_seconds or self.settings.signed_url_ttl_seconds)
        client = self._get_client()
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.settings.cloudflare_r2_bucket, "Key": object_key},
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
                    raise AuthorImageStoreError("author image exceeds maximum byte size")
            except ValueError:
                pass

        content = response.content
        if len(content) > self.settings.brand_asset_max_bytes:
            raise AuthorImageStoreError("author image exceeds maximum byte size")

        mime_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if not mime_type.startswith("image/"):
            raise AuthorImageStoreError(f"unsupported content type: {mime_type or 'unknown'}")

        return content, mime_type

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:
        try:
            import boto3
            from botocore.config import Config
        except Exception as exc:
            raise AuthorImageStoreConfigError(
                "boto3 is required for author image storage operations"
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
            raise AuthorImageStoreConfigError(
                "Missing required R2 config: " + ", ".join(sorted(missing))
            )
