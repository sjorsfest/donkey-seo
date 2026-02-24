"""Storage helpers for generated content featured images."""

from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Any

from app.config import Settings, settings


class ContentImageStoreError(RuntimeError):
    """Base error for generated content image storage operations."""


class ContentImageStoreConfigError(ContentImageStoreError):
    """Raised when required cloud storage settings are missing."""


class ContentImageStore:
    """Upload and retrieve generated featured images in private R2 storage."""

    def __init__(self, app_settings: Settings | None = None) -> None:
        self.settings = app_settings or settings
        self._client: Any | None = None

    async def upload_generated_image(
        self,
        *,
        project_id: str,
        brief_id: str,
        payload: bytes,
        source: str,
    ) -> dict[str, Any]:
        """Upload generated image payload and return immutable metadata."""
        self._validate_config()

        sha256 = hashlib.sha256(payload).hexdigest()
        object_key = f"projects/{project_id}/content-images/{brief_id}/{sha256}.png"
        width, height = self._detect_dimensions(payload)

        client = self._get_client()
        client.put_object(
            Bucket=self.settings.cloudflare_r2_bucket,
            Key=object_key,
            Body=payload,
            ContentType="image/png",
            Metadata={"sha256": sha256, "source": source},
        )

        return {
            "object_key": object_key,
            "mime_type": "image/png",
            "width": width,
            "height": height,
            "byte_size": len(payload),
            "sha256": sha256,
            "source": source,
        }

    def create_signed_read_url(self, *, object_key: str, ttl_seconds: int | None = None) -> str:
        """Mint short-lived signed URL for GET access to an object."""
        self._validate_config()
        expires_in = int(ttl_seconds or self.settings.signed_url_ttl_seconds)
        client = self._get_client()
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.settings.cloudflare_r2_bucket, "Key": object_key},
            ExpiresIn=expires_in,
        )

    def read_object_bytes(self, *, object_key: str) -> tuple[bytes, str | None]:
        """Read object bytes and content-type from private storage."""
        self._validate_config()
        client = self._get_client()
        response = client.get_object(
            Bucket=self.settings.cloudflare_r2_bucket,
            Key=object_key,
        )
        payload = response["Body"].read()
        mime_type = response.get("ContentType")
        return payload, str(mime_type) if mime_type else None

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:
        try:
            import boto3
            from botocore.config import Config
        except Exception as exc:
            raise ContentImageStoreConfigError(
                "boto3 is required for content image storage operations"
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

    def _validate_config(self) -> None:
        required = {
            "cloudflare_r2_account_id": self.settings.cloudflare_r2_account_id,
            "cloudflare_r2_access_key_id": self.settings.cloudflare_r2_access_key_id,
            "cloudflare_r2_secret_access_key": self.settings.cloudflare_r2_secret_access_key,
            "cloudflare_r2_bucket": self.settings.cloudflare_r2_bucket,
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise ContentImageStoreConfigError(
                "Missing required R2 config: " + ", ".join(sorted(missing))
            )

    @staticmethod
    def _detect_dimensions(payload: bytes) -> tuple[int | None, int | None]:
        try:
            from PIL import Image

            with Image.open(BytesIO(payload)) as image:
                width, height = image.size
                return int(width), int(height)
        except Exception:
            return None, None
