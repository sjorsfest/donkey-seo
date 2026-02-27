"""Resend email client for transactional messages."""

from __future__ import annotations

from typing import Any

import httpx

from app.config import settings
from app.core.exceptions import APIKeyMissingError, ExternalAPIError


class ResendEmailClient:
    """Minimal async Resend client."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.api_key = api_key or settings.resend_api_key
        self.timeout_seconds = timeout_seconds
        self._client: httpx.AsyncClient | None = None

        if not self.api_key:
            raise APIKeyMissingError("Resend")

    async def __aenter__(self) -> "ResendEmailClient":
        self._client = httpx.AsyncClient(
            timeout=self.timeout_seconds,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            base_url="https://api.resend.com",
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("ResendEmailClient must be used as async context manager")
        return self._client

    async def send_email(
        self,
        *,
        from_email: str,
        to_email: str,
        subject: str,
        html: str,
    ) -> dict[str, Any]:
        payload = {
            "from": from_email,
            "to": [to_email],
            "subject": subject,
            "html": html,
        }
        try:
            response = await self.client.post("/emails", json=payload)
        except httpx.HTTPError as exc:
            raise ExternalAPIError("Resend", str(exc)) from exc

        try:
            body = response.json()
        except ValueError as exc:
            raise ExternalAPIError(
                "Resend",
                f"Invalid JSON response with status {response.status_code}",
            ) from exc

        if response.status_code >= 400:
            message = "request_failed"
            if isinstance(body, dict) and isinstance(body.get("message"), str):
                message = body["message"]
            raise ExternalAPIError("Resend", message)

        if not isinstance(body, dict):
            raise ExternalAPIError("Resend", "Unexpected non-object response")
        return body
