"""User model for authentication."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, TypedModelMixin, UUIDMixin
from app.models.generated_dtos import UserCreateDTO, UserPatchDTO

if TYPE_CHECKING:
    from app.models.oauth_account import OAuthAccount
    from app.models.project import Project


class User(TypedModelMixin[UserCreateDTO, UserPatchDTO], Base, UUIDMixin, TimestampMixin):
    """User account for authentication and project ownership."""

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str | None] = mapped_column(String(255), nullable=True)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(default=False, nullable=False)
    stripe_customer_id: Mapped[str | None] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=True,
    )
    stripe_subscription_id: Mapped[str | None] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=True,
    )
    subscription_plan: Mapped[str | None] = mapped_column(String(50), nullable=True)
    subscription_interval: Mapped[str | None] = mapped_column(String(20), nullable=True)
    subscription_status: Mapped[str | None] = mapped_column(
        String(32),
        default="inactive",
        nullable=False,
        index=True,
    )
    subscription_current_period_end: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    subscription_trial_ends_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    stripe_price_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Relationships
    projects: Mapped[list[Project]] = relationship(
        "Project",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    oauth_accounts: Mapped[list[OAuthAccount]] = relationship(
        "OAuthAccount",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<User {self.email}>"
