"""JWT authentication and password hashing utilities."""

import logging

from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from jose import JWTError, jwt

from app.config import settings
from app.core.exceptions import InvalidTokenError

logger = logging.getLogger(__name__)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"), hashed_password.encode("utf-8")
    )


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return bcrypt.hashpw(
        password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")


def create_access_token(
    subject: str,
    expires_delta: timedelta | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Create a JWT access token."""
    logger.info("Creating access token", extra={"subject": subject})
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.access_token_expire_minutes
        )

    to_encode: dict[str, Any] = {
        "sub": subject,
        "exp": expire,
        "type": "access",
    }
    if extra_claims:
        to_encode.update(extra_claims)

    return jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def create_refresh_token(
    subject: str,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a JWT refresh token."""
    logger.info("Creating refresh token", extra={"subject": subject})
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            days=settings.refresh_token_expire_days
        )

    to_encode: dict[str, Any] = {
        "sub": subject,
        "exp": expire,
        "type": "refresh",
    }

    return jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def create_email_verification_token(
    subject: str,
    email: str,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a JWT email verification token."""
    logger.info("Creating email verification token", extra={"subject": subject, "email": email})
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            hours=settings.email_verification_token_expire_hours
        )

    to_encode: dict[str, Any] = {
        "sub": subject,
        "email": email,
        "exp": expire,
        "type": "email_verification",
    }

    return jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def decode_token(token: str, expected_type: str = "access") -> dict[str, Any]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )

        # Verify token type
        token_type = payload.get("type")
        if token_type != expected_type:
            logger.warning("Token type mismatch", extra={"expected": expected_type, "got": token_type})
            raise InvalidTokenError(f"Expected {expected_type} token, got {token_type}")

        # Check if subject exists
        if payload.get("sub") is None:
            logger.warning("Token missing subject")
            raise InvalidTokenError("Token missing subject")

        return payload

    except JWTError as e:
        logger.warning("Token decode failed", extra={"error": str(e)})
        raise InvalidTokenError(str(e)) from e


def verify_access_token(token: str) -> str:
    """Verify access token and return the subject (user_id)."""
    payload = decode_token(token, expected_type="access")
    return payload["sub"]


def verify_refresh_token(token: str) -> str:
    """Verify refresh token and return the subject (user_id)."""
    payload = decode_token(token, expected_type="refresh")
    return payload["sub"]


def verify_email_verification_token(token: str) -> tuple[str, str]:
    """Verify email verification token and return (user_id, email)."""
    payload = decode_token(token, expected_type="email_verification")
    email = payload.get("email")
    if not isinstance(email, str) or not email:
        logger.warning("Email verification token missing email claim")
        raise InvalidTokenError("Token missing email")
    return payload["sub"], email
