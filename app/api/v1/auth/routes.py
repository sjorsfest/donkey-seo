"""Authentication API endpoints."""

import hashlib
import logging
import secrets
import time
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from sqlalchemy import select

from app.api.v1.auth.constants import (
    GOOGLE_AUTH_URL,
    GOOGLE_SCOPES,
    STATE_TTL_SECONDS,
    TWITTER_AUTH_URL,
    TWITTER_SCOPES,
)
from app.api.v1.auth.utils import (
    b64url_encode,
    decode_oauth_state,
    default_frontend_redirect,
    encode_oauth_state,
    exchange_google_code_for_token,
    exchange_twitter_code_for_token,
    fetch_google_profile,
    fetch_twitter_profile,
    find_or_create_oauth_user,
    is_valid_redirect_uri,
    oauth_error_redirect,
    oauth_success_redirect,
    require_google_oauth_config,
    require_twitter_oauth_config,
)
from app.config import settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    get_password_hash,
    verify_password,
    verify_refresh_token,
)
from app.dependencies import CurrentUser, DbSession
from app.models.generated_dtos import UserCreateDTO
from app.models.user import User
from app.schemas.auth import Token, TokenRefresh, UserCreate, UserLogin, UserResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, session: DbSession) -> User:
    """Register a new user account."""
    logger.info("User registration attempt", extra={"email": user_data.email})

    result = await session.execute(select(User).where(User.email == user_data.email))
    existing_user = result.scalar_one_or_none()

    if existing_user:
        logger.warning("Registration failed: email exists", extra={"email": user_data.email})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists",
        )

    user = User.create(
        session,
        UserCreateDTO(
            email=user_data.email,
            hashed_password=get_password_hash(user_data.password),
            full_name=user_data.full_name,
        ),
    )
    await session.flush()
    await session.refresh(user)

    logger.info("User registered", extra={"user_id": str(user.id), "email": user.email})

    return user


@router.post("/login", response_model=Token)
async def login(credentials: UserLogin, session: DbSession) -> Token:
    """Login and get access/refresh tokens."""
    logger.info("Login attempt", extra={"email": credentials.email})

    result = await session.execute(select(User).where(User.email == credentials.email))
    user = result.scalar_one_or_none()

    if (
        not user
        or not user.hashed_password
        or not verify_password(credentials.password, user.hashed_password)
    ):
        logger.warning("Login failed: invalid credentials", extra={"email": credentials.email})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        logger.warning("Login failed: inactive user", extra={"email": credentials.email})
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated",
        )

    access_token = create_access_token(subject=str(user.id))
    refresh_token = create_refresh_token(subject=str(user.id))

    logger.info("Login successful", extra={"user_id": str(user.id)})

    return Token(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=Token)
async def refresh_token(token_data: TokenRefresh, session: DbSession) -> Token:
    """Refresh access token using refresh token."""
    try:
        user_id = verify_refresh_token(token_data.refresh_token)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Token refresh failed: invalid token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(subject=str(user.id))
    new_refresh_token = create_refresh_token(subject=str(user.id))

    return Token(access_token=access_token, refresh_token=new_refresh_token)


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: CurrentUser) -> User:
    """Get current user information."""
    return current_user


@router.get("/oauth/google/start")
async def oauth_google_start(redirect_uri: str = Query(..., min_length=1)) -> RedirectResponse:
    """Start Google OAuth flow and redirect to provider consent screen."""
    require_google_oauth_config()

    if not is_valid_redirect_uri(redirect_uri):
        raise HTTPException(status_code=400, detail="Invalid redirect_uri")

    state = encode_oauth_state(
        {
            "provider": "google",
            "redirect_uri": redirect_uri,
            "exp": int(time.time()) + STATE_TTL_SECONDS,
        }
    )

    authorize_params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_callback_url,
        "response_type": "code",
        "scope": GOOGLE_SCOPES,
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    provider_url = f"{GOOGLE_AUTH_URL}?{urlencode(authorize_params)}"
    return RedirectResponse(url=provider_url, status_code=status.HTTP_302_FOUND)


@router.get("/oauth/twitter/start")
async def oauth_twitter_start(redirect_uri: str = Query(..., min_length=1)) -> RedirectResponse:
    """Start Twitter OAuth flow and redirect to provider consent screen."""
    require_twitter_oauth_config()

    if not is_valid_redirect_uri(redirect_uri):
        raise HTTPException(status_code=400, detail="Invalid redirect_uri")

    code_verifier = secrets.token_urlsafe(64)
    code_challenge = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    code_challenge_b64 = b64url_encode(code_challenge)

    state = encode_oauth_state(
        {
            "provider": "twitter",
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
            "exp": int(time.time()) + STATE_TTL_SECONDS,
        }
    )

    authorize_params = {
        "client_id": settings.twitter_client_id,
        "redirect_uri": settings.twitter_callback_url,
        "response_type": "code",
        "scope": TWITTER_SCOPES,
        "state": state,
        "code_challenge": code_challenge_b64,
        "code_challenge_method": "S256",
    }
    provider_url = f"{TWITTER_AUTH_URL}?{urlencode(authorize_params)}"
    return RedirectResponse(url=provider_url, status_code=status.HTTP_302_FOUND)


@router.get("/oauth/google/callback")
async def oauth_google_callback(
    session: DbSession,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
) -> RedirectResponse:
    """Handle Google OAuth callback, issue local JWTs, and redirect to frontend."""
    redirect_uri = default_frontend_redirect()

    try:
        if state is None:
            raise ValueError("missing_state")
        state_payload = decode_oauth_state(state, expected_provider="google")
        redirect_uri = state_payload["redirect_uri"]
    except ValueError as exc:
        return oauth_error_redirect("google", redirect_uri, str(exc))

    if error:
        return oauth_error_redirect("google", redirect_uri, error)
    if not code:
        return oauth_error_redirect("google", redirect_uri, "missing_code")

    try:
        provider_access_token = await exchange_google_code_for_token(code)
        provider_user_id, email, full_name = await fetch_google_profile(provider_access_token)
        user = await find_or_create_oauth_user(
            session=session,
            provider="google",
            provider_user_id=provider_user_id,
            email=email,
            full_name=full_name,
        )
        if not user.is_active:
            return oauth_error_redirect("google", redirect_uri, "inactive_user")

        access_token = create_access_token(subject=str(user.id))
        refresh_token = create_refresh_token(subject=str(user.id))
        return oauth_success_redirect("google", redirect_uri, access_token, refresh_token)
    except ValueError as exc:
        logger.warning("Google OAuth callback failed", extra={"error": str(exc)})
        return oauth_error_redirect("google", redirect_uri, str(exc))
    except Exception:  # noqa: BLE001
        logger.exception("Google OAuth callback unexpected error")
        return oauth_error_redirect("google", redirect_uri, "oauth_failed")


@router.get("/oauth/twitter/callback")
async def oauth_twitter_callback(
    session: DbSession,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
) -> RedirectResponse:
    """Handle Twitter OAuth callback, issue local JWTs, and redirect to frontend."""
    redirect_uri = default_frontend_redirect()
    code_verifier = ""

    try:
        if state is None:
            raise ValueError("missing_state")
        state_payload = decode_oauth_state(state, expected_provider="twitter")
        redirect_uri = state_payload["redirect_uri"]
        code_verifier = state_payload.get("code_verifier", "")
        if not isinstance(code_verifier, str) or not code_verifier:
            raise ValueError("invalid_state")
    except ValueError as exc:
        return oauth_error_redirect("twitter", redirect_uri, str(exc))

    if error:
        return oauth_error_redirect("twitter", redirect_uri, error)
    if not code:
        return oauth_error_redirect("twitter", redirect_uri, "missing_code")

    try:
        provider_access_token = await exchange_twitter_code_for_token(code, code_verifier)
        provider_user_id, email, full_name = await fetch_twitter_profile(provider_access_token)
        user = await find_or_create_oauth_user(
            session=session,
            provider="twitter",
            provider_user_id=provider_user_id,
            email=email,
            full_name=full_name,
        )
        if not user.is_active:
            return oauth_error_redirect("twitter", redirect_uri, "inactive_user")

        access_token = create_access_token(subject=str(user.id))
        refresh_token = create_refresh_token(subject=str(user.id))
        return oauth_success_redirect("twitter", redirect_uri, access_token, refresh_token)
    except ValueError as exc:
        logger.warning("Twitter OAuth callback failed", extra={"error": str(exc)})
        return oauth_error_redirect("twitter", redirect_uri, str(exc))
    except Exception:  # noqa: BLE001
        logger.exception("Twitter OAuth callback unexpected error")
        return oauth_error_redirect("twitter", redirect_uri, "oauth_failed")
