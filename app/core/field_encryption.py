"""Field-level encryption helpers for secrets stored in the database."""

from __future__ import annotations

from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from app.config import settings


@lru_cache(maxsize=1)
def get_webhook_secret_cipher() -> Fernet:
    """Return Fernet cipher used for webhook-secret encryption/decryption."""
    return Fernet(settings.get_webhook_secret_encryption_key())


def reset_webhook_secret_cipher_cache() -> None:
    """Reset cached webhook secret cipher (useful in tests after key changes)."""
    get_webhook_secret_cipher.cache_clear()


def encrypt_webhook_secret(value: str) -> str:
    """Encrypt a webhook secret for persistent storage."""
    return get_webhook_secret_cipher().encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt_webhook_secret(value: str) -> str:
    """Decrypt a stored webhook secret ciphertext."""
    try:
        return get_webhook_secret_cipher().decrypt(value.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise ValueError("invalid_webhook_secret_ciphertext") from exc
