"""Unit tests for integration key utility helpers."""

import hashlib
import hmac

from app.config import settings
from app.core.integration_keys import (
    generate_integration_api_key,
    generate_webhook_secret,
    hash_integration_api_key,
)


def test_generate_integration_api_key_prefix(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.core.integration_keys.secrets.token_urlsafe",
        lambda _bytes: "abc123",
    )

    assert generate_integration_api_key() == "dseo_abc123"


def test_hash_integration_api_key_normalizes_whitespace() -> None:
    original_pepper = settings.integration_api_key_pepper
    settings.integration_api_key_pepper = "pepper-123"
    value = "  dseo_test_key  "

    try:
        assert hash_integration_api_key(value) == hmac.new(
            b"integration-api-key:pepper-123",
            b"dseo_test_key",
            hashlib.sha256,
        ).hexdigest()
    finally:
        settings.integration_api_key_pepper = original_pepper


def test_generate_webhook_secret_prefix(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.core.integration_keys.secrets.token_urlsafe",
        lambda _bytes: "secret-token",
    )

    assert generate_webhook_secret() == "whsec_secret-token"
