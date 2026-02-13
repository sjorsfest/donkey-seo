"""Facade for typed write operations."""

from __future__ import annotations

from typing import TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.typed.contracts import CreateDTOProtocol, PatchDTOProtocol
from app.persistence.typed.registry import get_adapter

ModelT = TypeVar("ModelT")
CreateDTOT = TypeVar("CreateDTOT", bound=CreateDTOProtocol)
PatchDTOT = TypeVar("PatchDTOT", bound=PatchDTOProtocol)


def create(
    session: AsyncSession,
    model_cls: type[ModelT],
    dto: CreateDTOT,
) -> ModelT:
    """Create model instance via registered adapter."""
    adapter = get_adapter(model_cls)
    return adapter.create(session, dto)


def patch(
    session: AsyncSession,
    model_cls: type[ModelT],
    instance: ModelT,
    dto: PatchDTOT,
) -> ModelT:
    """Patch model instance via registered adapter."""
    adapter = get_adapter(model_cls)
    return adapter.patch(session, instance, dto)


async def delete(
    session: AsyncSession,
    model_cls: type[ModelT],
    instance: ModelT,
) -> None:
    """Delete model instance via registered adapter."""
    adapter = get_adapter(model_cls)
    await adapter.delete(session, instance)
