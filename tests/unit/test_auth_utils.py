"""Unit tests for auth utility helpers."""

from app.api.v1.auth.utils import build_email_verification_url, sanitize_auth_email
from app.config import settings


def test_sanitize_auth_email_strips_plus_alias() -> None:
    assert sanitize_auth_email("Foo+test@Bar.com") == "foo@bar.com"


def test_sanitize_auth_email_keeps_address_without_plus() -> None:
    assert sanitize_auth_email(" User@Example.com ") == "user@example.com"


def test_sanitize_auth_email_keeps_non_email_like_values_stable() -> None:
    assert sanitize_auth_email("not-an-email") == "not-an-email"


def test_build_email_verification_url_uses_explicit_callback() -> None:
    original_callback = settings.email_verification_callback_url
    token = "token-123"
    settings.email_verification_callback_url = "https://donkeyseo.io/verify-email"
    try:
        url = build_email_verification_url(token)
    finally:
        settings.email_verification_callback_url = original_callback

    assert url == "https://donkeyseo.io/verify-email?token=token-123"


def test_build_email_verification_url_uses_public_api_base_fallback() -> None:
    original_callback = settings.email_verification_callback_url
    original_public_base = settings.public_api_base_url
    settings.email_verification_callback_url = None
    settings.public_api_base_url = "https://api.donkeyseo.io"
    try:
        url = build_email_verification_url("abc")
    finally:
        settings.email_verification_callback_url = original_callback
        settings.public_api_base_url = original_public_base

    assert url == "https://api.donkeyseo.io/api/v1/auth/verify-email?token=abc"
