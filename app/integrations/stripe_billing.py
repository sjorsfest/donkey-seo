"""Stripe billing client for subscriptions and webhook verification."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from app.config import settings
from app.core.exceptions import APIKeyMissingError, ExternalAPIError


class StripeSignatureError(ValueError):
    """Raised when Stripe webhook signature verification fails."""


class StripeBillingClient:
    """Minimal async Stripe client for billing workflows."""

    BASE_URL = "https://api.stripe.com/v1"

    def __init__(
        self,
        *,
        secret_key: str | None = None,
        webhook_secret: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.secret_key = secret_key or settings.stripe_secret_key
        self.webhook_secret = webhook_secret or settings.stripe_webhook_secret
        self.timeout_seconds = timeout_seconds
        self._client: httpx.AsyncClient | None = None

        if not self.secret_key:
            raise APIKeyMissingError("Stripe")

    async def __aenter__(self) -> "StripeBillingClient":
        self._client = httpx.AsyncClient(
            timeout=self.timeout_seconds,
            headers={
                "Authorization": f"Bearer {self.secret_key}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            base_url=self.BASE_URL,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("StripeBillingClient must be used as async context manager")
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        *,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = await self.client.request(
                method,
                path,
                data=self._to_form_fields(data),
                params=params,
            )
        except httpx.HTTPError as exc:
            raise ExternalAPIError("Stripe", str(exc)) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise ExternalAPIError(
                "Stripe",
                f"Invalid JSON response with status {response.status_code}",
            ) from exc

        if response.status_code >= 400:
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict):
                message = str(error.get("message") or "Unknown Stripe error")
                code = str(error.get("code") or "unknown")
                param = str(error.get("param") or "unknown")
                raise ExternalAPIError(
                    "Stripe",
                    f"{message} (code={code}, param={param})",
                )
            raise ExternalAPIError("Stripe", f"HTTP {response.status_code}")

        if not isinstance(payload, dict):
            raise ExternalAPIError("Stripe", "Unexpected non-object response")
        return payload

    @staticmethod
    def _to_form_fields(payload: dict[str, Any] | None) -> dict[str, str]:
        if not payload:
            return {}
        return {str(key): str(value) for key, value in payload.items() if value is not None}

    async def create_customer(
        self,
        *,
        email: str,
        full_name: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "email": email,
            "name": full_name or "",
        }
        for key, value in (metadata or {}).items():
            data[f"metadata[{key}]"] = value
        return await self._request("POST", "/customers", data=data)

    async def retrieve_subscription(self, subscription_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/subscriptions/{subscription_id}")

    async def retrieve_price(self, price_id: str) -> dict[str, Any]:
        return await self._request(
            "GET",
            f"/prices/{price_id}",
            params={"expand[]": "product"},
        )

    async def create_checkout_session(
        self,
        *,
        customer_id: str,
        price_id: str,
        success_url: str,
        cancel_url: str,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "mode": "subscription",
            "customer": customer_id,
            "line_items[0][price]": price_id,
            "line_items[0][quantity]": 1,
            "success_url": success_url,
            "cancel_url": cancel_url,
            "allow_promotion_codes": "true",
        }
        return await self._request("POST", "/checkout/sessions", data=data)

    async def create_billing_portal_session(
        self,
        *,
        customer_id: str,
        return_url: str,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "customer": customer_id,
            "return_url": return_url,
        }
        return await self._request("POST", "/billing_portal/sessions", data=data)

    def verify_webhook_event(
        self,
        *,
        payload: bytes,
        signature_header: str,
        tolerance_seconds: int = 300,
    ) -> dict[str, Any]:
        """Verify Stripe webhook signature and return decoded event payload."""
        if not self.webhook_secret:
            raise APIKeyMissingError("Stripe webhook signing secret")

        timestamp, signatures = self._parse_signature_header(signature_header)
        expected = hmac.new(
            self.webhook_secret.encode("utf-8"),
            msg=f"{timestamp}.{payload.decode('utf-8')}".encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()

        if not any(hmac.compare_digest(expected, sig) for sig in signatures):
            raise StripeSignatureError("Invalid Stripe signature")

        now_ts = int(time.time())
        if abs(now_ts - timestamp) > tolerance_seconds:
            raise StripeSignatureError("Stripe signature timestamp outside tolerance")

        try:
            event_payload = json.loads(payload.decode("utf-8"))
        except ValueError as exc:
            raise StripeSignatureError("Invalid JSON payload") from exc

        if not isinstance(event_payload, dict):
            raise StripeSignatureError("Invalid Stripe event object")
        return event_payload

    @staticmethod
    def _parse_signature_header(header_value: str) -> tuple[int, list[str]]:
        timestamp: int | None = None
        signatures: list[str] = []
        for part in header_value.split(","):
            key, sep, value = part.partition("=")
            if sep != "=":
                continue
            if key == "t":
                try:
                    timestamp = int(value)
                except ValueError as exc:
                    raise StripeSignatureError("Invalid Stripe signature timestamp") from exc
            elif key == "v1":
                signatures.append(value)

        if timestamp is None or not signatures:
            raise StripeSignatureError("Malformed Stripe signature header")
        return timestamp, signatures


def stripe_unix_to_datetime(value: Any) -> datetime | None:
    """Convert Stripe unix timestamp value into timezone-aware datetime."""
    if value is None:
        return None
    try:
        ts = int(value)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc)
