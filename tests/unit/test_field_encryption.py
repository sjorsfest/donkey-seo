"""Unit tests for encrypted webhook secret helpers and column type."""

from __future__ import annotations

from app.config import settings
from app.core.field_encryption import (
    decrypt_webhook_secret,
    encrypt_webhook_secret,
    reset_webhook_secret_cipher_cache,
)
from app.models.base import EncryptedString


def test_webhook_secret_encrypt_decrypt_roundtrip() -> None:
    original_key = settings.webhook_secret_encryption_key
    settings.webhook_secret_encryption_key = "unit-test-key"
    reset_webhook_secret_cipher_cache()

    try:
        encrypted = encrypt_webhook_secret("secret-123")
        assert encrypted != "secret-123"
        assert decrypt_webhook_secret(encrypted) == "secret-123"
    finally:
        settings.webhook_secret_encryption_key = original_key
        reset_webhook_secret_cipher_cache()


def test_encrypted_string_type_encrypts_and_decrypts_values() -> None:
    original_key = settings.webhook_secret_encryption_key
    settings.webhook_secret_encryption_key = "unit-test-key"
    reset_webhook_secret_cipher_cache()

    try:
        encrypted_type = EncryptedString(length=1024)
        encrypted = encrypted_type.process_bind_param("my-webhook-secret", dialect=None)
        assert isinstance(encrypted, str)
        assert encrypted != "my-webhook-secret"

        decrypted = encrypted_type.process_result_value(encrypted, dialect=None)
        assert decrypted == "my-webhook-secret"
        assert encrypted_type.process_bind_param("   ", dialect=None) is None
    finally:
        settings.webhook_secret_encryption_key = original_key
        reset_webhook_secret_cipher_cache()
