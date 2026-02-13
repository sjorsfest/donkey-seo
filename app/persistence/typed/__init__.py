"""Typed write operations for SQLAlchemy models."""

from app.persistence.typed.writes import create, delete, patch

__all__ = ["create", "patch", "delete"]
