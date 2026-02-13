"""Typed write layer errors."""

from __future__ import annotations


class TypedWriteError(RuntimeError):
    """Base error for typed write layer."""


class AdapterNotFoundError(TypedWriteError):
    """Raised when no write adapter is registered for a model."""

    def __init__(self, model_name: str) -> None:
        super().__init__(f"No typed write adapter registered for model: {model_name}")


class InvalidPatchFieldError(TypedWriteError):
    """Raised when a patch payload contains disallowed fields."""

    def __init__(self, model_name: str, fields: list[str]) -> None:
        invalid = ", ".join(sorted(fields))
        super().__init__(f"Invalid patch fields for {model_name}: {invalid}")
