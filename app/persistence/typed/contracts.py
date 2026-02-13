"""Contracts for typed SQLAlchemy writes."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from sqlalchemy.ext.asyncio import AsyncSession

ModelT = TypeVar("ModelT")
CreateDTOT = TypeVar("CreateDTOT", bound="CreateDTOProtocol")
PatchDTOT = TypeVar("PatchDTOT", bound="PatchDTOProtocol")


@runtime_checkable
class CreateDTOProtocol(Protocol):
    """Protocol for generated create DTOs."""

    def to_orm_kwargs(self) -> dict[str, Any]:
        """Convert DTO to ORM constructor payload."""


@runtime_checkable
class PatchDTOProtocol(Protocol):
    """Protocol for generated patch DTOs."""

    def to_patch_dict(self) -> dict[str, Any]:
        """Convert DTO to sparse patch payload."""


class CreatePort(Protocol[ModelT, CreateDTOT]):
    """Port for typed model creation."""

    def create(self, session: AsyncSession, dto: CreateDTOT) -> ModelT:
        """Create and add a model instance to the session."""


class PatchPort(Protocol[ModelT, PatchDTOT]):
    """Port for typed model patching."""

    def patch(self, session: AsyncSession, instance: ModelT, dto: PatchDTOT) -> ModelT:
        """Patch an existing model instance."""


class DeletePort(Protocol[ModelT]):
    """Port for typed model deletion."""

    async def delete(self, session: AsyncSession, instance: ModelT) -> None:
        """Delete a model instance from the session."""


class WriteAdapter(
    CreatePort[ModelT, CreateDTOT],
    PatchPort[ModelT, PatchDTOT],
    DeletePort[ModelT],
    Protocol[ModelT, CreateDTOT, PatchDTOT],
):
    """Combined adapter contract for typed writes."""

    model_cls: type[ModelT]
