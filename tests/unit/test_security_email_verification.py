"""Unit tests for email verification security helpers."""

import pytest

from app.core.exceptions import InvalidTokenError
from app.core.security import (
    create_access_token,
    create_email_verification_token,
    verify_email_verification_token,
)


def test_email_verification_token_roundtrip() -> None:
    token = create_email_verification_token(
        subject="user_123",
        email="user@example.com",
    )
    user_id, email = verify_email_verification_token(token)

    assert user_id == "user_123"
    assert email == "user@example.com"


def test_email_verification_token_rejects_wrong_token_type() -> None:
    token = create_access_token(subject="user_123")

    with pytest.raises(InvalidTokenError):
        verify_email_verification_token(token)
