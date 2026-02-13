"""Base adapter implementation for typed SQLAlchemy writes."""

from __future__ import annotations

import uuid
from typing import Any, Generic, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.typed.contracts import (
    CreateDTOProtocol,
    PatchDTOProtocol,
)
from app.persistence.typed.errors import InvalidPatchFieldError

ModelT = TypeVar("ModelT")
CreateDTOT = TypeVar("CreateDTOT", bound=CreateDTOProtocol)
PatchDTOT = TypeVar("PatchDTOT", bound=PatchDTOProtocol)


class BaseWriteAdapter(Generic[ModelT, CreateDTOT, PatchDTOT]):
    """Default write adapter with strict patch-field validation."""

    model_cls: type[ModelT]
    patch_allowlist: set[str]

    def __init__(self, model_cls: type[ModelT], patch_allowlist: set[str]) -> None:
        self.model_cls = model_cls
        self.patch_allowlist = patch_allowlist

    def create(self, session: AsyncSession, dto: CreateDTOT) -> ModelT:
        """Create model from DTO and add it to the session."""
        payload = self._normalize_payload(dto.to_orm_kwargs())
        instance = self.model_cls(**payload)
        session.add(instance)
        return instance

    def patch(self, session: AsyncSession, instance: ModelT, dto: PatchDTOT) -> ModelT:
        """Patch model fields from sparse DTO payload."""
        payload = self._normalize_payload(dto.to_patch_dict())
        invalid_fields = sorted(set(payload.keys()) - self.patch_allowlist)
        if invalid_fields:
            raise InvalidPatchFieldError(self.model_cls.__name__, invalid_fields)

        for key, value in payload.items():
            setattr(instance, key, value)

        return instance

    async def delete(self, session: AsyncSession, instance: ModelT) -> None:
        """Delete model instance from session."""
        await session.delete(instance)

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Normalize payload values (UUID -> str) recursively."""

        def _normalize(value: Any) -> Any:
            if isinstance(value, uuid.UUID):
                return str(value)
            if isinstance(value, list):
                return [_normalize(item) for item in value]
            if isinstance(value, dict):
                return {k: _normalize(v) for k, v in value.items()}
            return value

        return {key: _normalize(value) for key, value in payload.items()}
