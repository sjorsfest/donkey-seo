"""Utilities for integration API keys and webhook secrets."""

from __future__ import annotations

import hashlib
import secrets

INTEGRATION_API_KEY_PREFIX = "dseo_"
WEBHOOK_SECRET_PREFIX = "whsec_"


def generate_integration_api_key(token_bytes: int = 32) -> str:
    """Generate a plaintext integration API key."""
    return f"{INTEGRATION_API_KEY_PREFIX}{secrets.token_urlsafe(token_bytes)}"


def hash_integration_api_key(api_key: str) -> str:
    """Hash an integration API key for storage and lookup."""
    normalized = api_key.strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def generate_webhook_secret(token_bytes: int = 32) -> str:
    """Generate a webhook signing secret."""
    return f"{WEBHOOK_SECRET_PREFIX}{secrets.token_urlsafe(token_bytes)}"
