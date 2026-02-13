"""Utility helpers for authentication and OAuth routes."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
from fastapi import HTTPException, status
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.auth.constants import (
    GOOGLE_TOKEN_URL,
    GOOGLE_USERINFO_URL,
    TWITTER_TOKEN_URL,
    TWITTER_USERINFO_URL,
)
from app.config import settings
from app.models.generated_dtos import (
    OAuthAccountCreateDTO,
    OAuthAccountPatchDTO,
    UserCreateDTO,
    UserPatchDTO,
)
from app.models.oauth_account import OAuthAccount
from app.models.user import User

logger = logging.getLogger(__name__)


def is_valid_redirect_uri(redirect_uri: str) -> bool:
    """Validate that redirect URI is an absolute http(s) URL."""
    parsed = urlparse(redirect_uri)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def default_frontend_redirect() -> str:
    """Fallback redirect target for callback failures."""
    if settings.cors_origins:
        return settings.cors_origins[0]
    return "http://localhost:3000"


def _b64url_encode(data: bytes) -> str:
    """Base64url encode without padding."""
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def b64url_encode(data: bytes) -> str:
    """Public helper for base64url encoding without padding."""
    return _b64url_encode(data)


def _b64url_decode(data: str) -> bytes:
    """Base64url decode with padding restoration."""
    padding = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + padding)


def encode_oauth_state(payload: dict[str, Any]) -> str:
    """Sign and encode OAuth state payload."""
    if not settings.oauth_state_secret:
        raise ValueError("oauth_state_secret_not_configured")

    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    signature = hmac.new(
        settings.oauth_state_secret.encode("utf-8"),
        payload_json,
        hashlib.sha256,
    ).digest()
    return f"{_b64url_encode(payload_json)}.{_b64url_encode(signature)}"


def decode_oauth_state(state: str, expected_provider: str) -> dict[str, Any]:
    """Verify and decode OAuth state payload."""
    try:
        encoded_payload, encoded_signature = state.split(".", 1)
        payload_bytes = _b64url_decode(encoded_payload)
        signature_bytes = _b64url_decode(encoded_signature)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("invalid_state") from exc

    expected_signature = hmac.new(
        settings.oauth_state_secret.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(signature_bytes, expected_signature):
        raise ValueError("invalid_state")

    try:
        decoded_payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError("invalid_state") from exc

    if not isinstance(decoded_payload, dict):
        raise ValueError("invalid_state")
    payload: dict[str, Any] = {str(key): value for key, value in decoded_payload.items()}

    if payload.get("provider") != expected_provider:
        raise ValueError("invalid_state")

    expires_at = int(payload.get("exp", 0))
    if expires_at < int(time.time()):
        raise ValueError("state_expired")

    redirect_uri = payload.get("redirect_uri")
    if not isinstance(redirect_uri, str) or not is_valid_redirect_uri(redirect_uri):
        raise ValueError("invalid_redirect_uri")

    return payload


def build_redirect_url(base_uri: str, params: dict[str, str]) -> str:
    """Append query parameters to a URL (overwrites duplicate keys)."""
    parsed = urlparse(base_uri)
    current_params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    current_params.update(params)
    new_query = urlencode(current_params)
    return urlunparse(parsed._replace(query=new_query))


def oauth_error_redirect(provider: str, redirect_uri: str, error: str) -> RedirectResponse:
    """Build redirect response with OAuth error information."""
    target = build_redirect_url(
        redirect_uri,
        {
            "provider": provider,
            "error": error,
        },
    )
    return RedirectResponse(url=target, status_code=status.HTTP_302_FOUND)


def oauth_success_redirect(
    provider: str,
    redirect_uri: str,
    access_token: str,
    refresh_token: str,
) -> RedirectResponse:
    """Build redirect response with issued local tokens."""
    target = build_redirect_url(
        redirect_uri,
        {
            "provider": provider,
            "access_token": access_token,
            "refresh_token": refresh_token,
        },
    )
    return RedirectResponse(url=target, status_code=status.HTTP_302_FOUND)


def _build_social_placeholder_email(provider: str, provider_user_id: str) -> str:
    """Generate deterministic placeholder email for providers without email scope."""
    normalized_user_id = "".join(ch for ch in provider_user_id.lower() if ch.isalnum())
    normalized_user_id = normalized_user_id[:64] or secrets.token_hex(8)
    return f"{provider}-{normalized_user_id}@oauth.local"


async def _ensure_unique_email(session: AsyncSession, base_email: str) -> str:
    """Ensure generated email is unique in users table."""
    local_part, domain_part = base_email.split("@", 1)
    candidate = base_email
    suffix = 1

    while True:
        result = await session.execute(select(User).where(User.email == candidate))
        if result.scalar_one_or_none() is None:
            return candidate
        candidate = f"{local_part}+{suffix}@{domain_part}"
        suffix += 1


async def find_or_create_oauth_user(
    session: AsyncSession,
    provider: str,
    provider_user_id: str,
    email: str | None,
    full_name: str | None,
) -> User:
    """Find or create a local user from provider identity."""
    oauth_result = await session.execute(
        select(OAuthAccount).where(
            OAuthAccount.provider == provider,
            OAuthAccount.provider_user_id == provider_user_id,
        )
    )
    oauth_account = oauth_result.scalar_one_or_none()

    if oauth_account:
        user_result = await session.execute(select(User).where(User.id == oauth_account.user_id))
        linked_user = user_result.scalar_one_or_none()
        if linked_user is None:
            raise ValueError("oauth_account_user_missing")

        if email and oauth_account.email != email:
            oauth_account.patch(
                session,
                OAuthAccountPatchDTO.from_partial({"email": email}),
            )
        if full_name and not linked_user.full_name:
            linked_user.patch(
                session,
                UserPatchDTO.from_partial({"full_name": full_name}),
            )

        await session.flush()
        return linked_user

    user: User | None = None
    if email:
        user_result = await session.execute(select(User).where(User.email == email))
        user = user_result.scalar_one_or_none()

    if user is None:
        user_email = email or _build_social_placeholder_email(provider, provider_user_id)
        if email is None:
            user_email = await _ensure_unique_email(session, user_email)

        user = User.create(
            session,
            UserCreateDTO(
                email=user_email,
                hashed_password=None,
                full_name=full_name,
            ),
        )
        await session.flush()
    elif full_name and not user.full_name:
        user.patch(
            session,
            UserPatchDTO.from_partial({"full_name": full_name}),
        )

    OAuthAccount.create(
        session,
        OAuthAccountCreateDTO(
            user_id=user.id,
            provider=provider,
            provider_user_id=provider_user_id,
            email=email,
        ),
    )
    await session.flush()
    await session.refresh(user)
    return user


def require_google_oauth_config() -> None:
    """Validate required Google OAuth settings are present."""
    if not settings.google_client_id:
        raise HTTPException(status_code=500, detail="GOOGLE_CLIENT_ID is not configured")
    if not settings.google_client_secret:
        raise HTTPException(status_code=500, detail="GOOGLE_CLIENT_SECRET is not configured")
    if not settings.google_callback_url:
        raise HTTPException(status_code=500, detail="GOOGLE_CALLBACK_URL is not configured")


def require_twitter_oauth_config() -> None:
    """Validate required Twitter OAuth settings are present."""
    if not settings.twitter_client_id:
        raise HTTPException(status_code=500, detail="TWITTER_CLIENT_ID is not configured")
    if not settings.twitter_client_secret:
        raise HTTPException(status_code=500, detail="TWITTER_CLIENT_SECRET is not configured")
    if not settings.twitter_callback_url:
        raise HTTPException(status_code=500, detail="TWITTER_CALLBACK_URL is not configured")


async def exchange_google_code_for_token(code: str) -> str:
    """Exchange Google authorization code for provider access token."""
    require_google_oauth_config()
    payload = {
        "code": code,
        "client_id": settings.google_client_id,
        "client_secret": settings.google_client_secret,
        "redirect_uri": settings.google_callback_url,
        "grant_type": "authorization_code",
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(GOOGLE_TOKEN_URL, data=payload)

    if response.status_code >= 400:
        logger.warning("Google token exchange failed", extra={"status_code": response.status_code})
        raise ValueError("google_token_exchange_failed")

    body = response.json()
    token = body.get("access_token")
    if not isinstance(token, str) or not token:
        raise ValueError("google_token_missing")
    return token


async def fetch_google_profile(provider_access_token: str) -> tuple[str, str | None, str | None]:
    """Fetch Google user profile fields needed for local account mapping."""
    headers = {"Authorization": f"Bearer {provider_access_token}"}

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(GOOGLE_USERINFO_URL, headers=headers)

    if response.status_code >= 400:
        logger.warning("Google profile fetch failed", extra={"status_code": response.status_code})
        raise ValueError("google_profile_fetch_failed")

    body = response.json()
    provider_user_id = body.get("sub")
    if not isinstance(provider_user_id, str) or not provider_user_id:
        raise ValueError("google_profile_missing_sub")

    email = body.get("email") if isinstance(body.get("email"), str) else None
    full_name = body.get("name") if isinstance(body.get("name"), str) else None
    return provider_user_id, email, full_name


async def exchange_twitter_code_for_token(code: str, code_verifier: str) -> str:
    """Exchange Twitter authorization code for provider access token."""
    require_twitter_oauth_config()
    client_id = settings.twitter_client_id
    client_secret = settings.twitter_client_secret
    callback_url = settings.twitter_callback_url
    if not client_id or not client_secret or not callback_url:
        raise ValueError("twitter_oauth_not_configured")

    payload = {
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": callback_url,
        "code_verifier": code_verifier,
        "client_id": client_id,
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            TWITTER_TOKEN_URL,
            data=payload,
            auth=(client_id, client_secret),
        )

    if response.status_code >= 400:
        logger.warning("Twitter token exchange failed", extra={"status_code": response.status_code})
        raise ValueError("twitter_token_exchange_failed")

    body = response.json()
    token = body.get("access_token")
    if not isinstance(token, str) or not token:
        raise ValueError("twitter_token_missing")
    return token


async def fetch_twitter_profile(provider_access_token: str) -> tuple[str, str | None, str | None]:
    """Fetch Twitter user profile fields needed for local account mapping."""
    headers = {"Authorization": f"Bearer {provider_access_token}"}
    params = {"user.fields": "name,username"}

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(TWITTER_USERINFO_URL, headers=headers, params=params)

    if response.status_code >= 400:
        logger.warning("Twitter profile fetch failed", extra={"status_code": response.status_code})
        raise ValueError("twitter_profile_fetch_failed")

    body = response.json()
    user_data = body.get("data")
    if not isinstance(user_data, dict):
        raise ValueError("twitter_profile_missing_data")

    provider_user_id = user_data.get("id")
    if not isinstance(provider_user_id, str) or not provider_user_id:
        raise ValueError("twitter_profile_missing_id")

    full_name = user_data.get("name") if isinstance(user_data.get("name"), str) else None
    username = user_data.get("username") if isinstance(user_data.get("username"), str) else None
    if full_name is None and username:
        full_name = username

    return provider_user_id, None, full_name
