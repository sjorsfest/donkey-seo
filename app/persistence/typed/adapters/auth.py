"""Authentication-related write adapters."""

from __future__ import annotations

from app.models.generated_dtos import (
    OAuthAccountCreateDTO,
    OAuthAccountPatchDTO,
    UserCreateDTO,
    UserPatchDTO,
)
from app.models.oauth_account import OAuthAccount
from app.models.user import User
from app.persistence.typed.adapters._base import BaseWriteAdapter

USER_PATCH_ALLOWLIST = {
    "email",
    "hashed_password",
    "full_name",
    "is_active",
    "is_superuser",
}

OAUTH_ACCOUNT_PATCH_ALLOWLIST = {
    "email",
}

_USER_ADAPTER = BaseWriteAdapter[User, UserCreateDTO, UserPatchDTO](
    model_cls=User,
    patch_allowlist=USER_PATCH_ALLOWLIST,
)

_OAUTH_ACCOUNT_ADAPTER = BaseWriteAdapter[
    OAuthAccount,
    OAuthAccountCreateDTO,
    OAuthAccountPatchDTO,
](
    model_cls=OAuthAccount,
    patch_allowlist=OAUTH_ACCOUNT_PATCH_ALLOWLIST,
)


def register() -> None:
    """Register auth adapters."""
    from app.persistence.typed.registry import register_adapter

    register_adapter(User, _USER_ADAPTER)
    register_adapter(OAuthAccount, _OAUTH_ACCOUNT_ADAPTER)
