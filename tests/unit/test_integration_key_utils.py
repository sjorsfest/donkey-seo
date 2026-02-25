"""Unit tests for integration key utility helpers."""

import hashlib

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
    value = "  dseo_test_key  "

    assert hash_integration_api_key(value) == hashlib.sha256(
        "dseo_test_key".encode("utf-8")
    ).hexdigest()


def test_generate_webhook_secret_prefix(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.core.integration_keys.secrets.token_urlsafe",
        lambda _bytes: "secret-token",
    )

    assert generate_webhook_secret() == "whsec_secret-token"
