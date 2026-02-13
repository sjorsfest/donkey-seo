"""OAuth account model for social login providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, StringUUID, TimestampMixin, TypedModelMixin, UUIDMixin
from app.models.generated_dtos import OAuthAccountCreateDTO, OAuthAccountPatchDTO

if TYPE_CHECKING:
    from app.models.user import User


class OAuthAccount(
    TypedModelMixin[OAuthAccountCreateDTO, OAuthAccountPatchDTO],
    Base,
    UUIDMixin,
    TimestampMixin,
):
    """External OAuth identity linked to a local user."""

    __tablename__ = "oauth_accounts"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "provider_user_id",
            name="uq_oauth_accounts_provider_user_id",
        ),
    )

    user_id: Mapped[str] = mapped_column(
        StringUUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    provider: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    provider_user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)

    user: Mapped[User] = relationship("User", back_populates="oauth_accounts")

    def __repr__(self) -> str:
        return f"<OAuthAccount {self.provider}:{self.provider_user_id}>"
