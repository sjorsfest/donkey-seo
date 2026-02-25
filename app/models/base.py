"""Base model and mixins for SQLAlchemy models."""

from datetime import datetime
from typing import Any, Generic, TypeVar

from sqlalchemy import DateTime, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import String, TypeDecorator

from app.core.ids import generate_cuid
from app.persistence.typed.contracts import CreateDTOProtocol, PatchDTOProtocol

CreateDTOT = TypeVar("CreateDTOT", bound=CreateDTOProtocol)
PatchDTOT = TypeVar("PatchDTOT", bound=PatchDTOProtocol)
ModelSelfT = TypeVar("ModelSelfT", bound="TypedModelMixin[Any, Any]")


class StringUUID(TypeDecorator):
    """String identifier type for CUID values."""

    impl = String(32)
    cache_ok = True

    def process_result_value(self, value, dialect):
        if value is not None:
            return str(value)
        return value

    def process_bind_param(self, value, dialect):
        if value is not None:
            return str(value)
        return value


class EncryptedString(TypeDecorator):
    """Encrypted string type using Fernet symmetric encryption."""

    impl = String
    cache_ok = False

    def __init__(self, length: int = 1024) -> None:
        super().__init__(length=length)

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        normalized = str(value).strip()
        if not normalized:
            return None

        from app.core.field_encryption import encrypt_webhook_secret

        return encrypt_webhook_secret(normalized)

    def process_result_value(self, value, dialect):
        if value is None:
            return None

        from app.core.field_encryption import decrypt_webhook_secret

        return decrypt_webhook_secret(str(value))


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class TypedModelMixin(Generic[CreateDTOT, PatchDTOT]):
    """Typed CRUD helpers delegated to the typed write adapter layer."""

    @classmethod
    async def get(
        cls: type[ModelSelfT],
        session: AsyncSession,
        model_id: str,
    ) -> ModelSelfT | None:
        """Fetch a model by primary key."""
        return await session.get(cls, model_id)

    @classmethod
    def create(cls: type[ModelSelfT], session: AsyncSession, dto: CreateDTOT) -> ModelSelfT:
        """Create a model via registered typed adapter."""
        from app.persistence.typed import create as typed_create

        return typed_create(session, cls, dto)

    def patch(self: ModelSelfT, session: AsyncSession, dto: PatchDTOT) -> ModelSelfT:
        """Patch a model via registered typed adapter."""
        from app.persistence.typed import patch as typed_patch

        return typed_patch(session, type(self), self, dto)

    async def delete(self, session: AsyncSession) -> None:
        """Delete a model via registered typed adapter."""
        from app.persistence.typed import delete as typed_delete

        await typed_delete(session, type(self), self)


class UUIDMixin:
    """Mixin that adds a CUID-style string primary key."""

    id: Mapped[str] = mapped_column(
        StringUUID(),
        primary_key=True,
        default=generate_cuid,
    )


class TimestampMixin:
    """Mixin that adds created_at and updated_at timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
