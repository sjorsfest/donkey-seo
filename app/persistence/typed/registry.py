"""Adapter registry for typed writes."""

from __future__ import annotations

from typing import Any

from app.persistence.typed.errors import AdapterNotFoundError

_ADAPTERS: dict[type[Any], Any] = {}
_INITIALIZED = False


def register_adapter(model_cls: type[Any], adapter: Any) -> None:
    """Register a write adapter for a model class."""
    _ADAPTERS[model_cls] = adapter


def _ensure_initialized() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    from app.persistence.typed.adapters import auth, content, keyword, pipeline, project, topic

    auth.register()
    content.register()
    keyword.register()
    pipeline.register()
    project.register()
    topic.register()

    _INITIALIZED = True


def get_adapter(model_cls: type[Any]) -> Any:
    """Resolve adapter for model class."""
    _ensure_initialized()
    adapter = _ADAPTERS.get(model_cls)
    if adapter is None:
        raise AdapterNotFoundError(model_cls.__name__)
    return adapter


def clear_registry() -> None:
    """Clear registry (used in tests)."""
    global _INITIALIZED
    _ADAPTERS.clear()
    _INITIALIZED = False
