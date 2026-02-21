"""Retry helpers for transient database connection failures."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeVar

from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError

logger = logging.getLogger(__name__)

_ResultT = TypeVar("_ResultT")

_CONNECTION_ERROR_MARKERS = (
    "connection is closed",
    "underlying connection is closed",
    "server closed the connection unexpectedly",
    "connection was closed",
)


def is_transient_connection_error(exc: Exception) -> bool:
    """Return True when an exception likely came from a dropped DB connection."""
    if isinstance(exc, (InterfaceError, OperationalError)):
        return True
    if isinstance(exc, DBAPIError) and exc.connection_invalidated:
        return True

    lowered = str(exc).lower()
    return any(marker in lowered for marker in _CONNECTION_ERROR_MARKERS)


async def run_with_transient_db_retry(
    operation: Callable[[], Awaitable[_ResultT]],
    *,
    operation_name: str,
    attempts: int = 3,
    base_delay_seconds: float = 0.2,
    log_context: Mapping[str, Any] | None = None,
) -> _ResultT:
    """Run an async DB operation and retry on transient connection failures."""
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    context = dict(log_context or {})
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except Exception as exc:
            if not is_transient_connection_error(exc) or attempt == attempts:
                raise
            logger.warning(
                "Transient database connection error; retrying operation",
                extra={
                    **context,
                    "operation": operation_name,
                    "attempt": attempt,
                    "max_attempts": attempts,
                },
            )
            await asyncio.sleep(base_delay_seconds * attempt)

    raise RuntimeError(f"Retry loop exhausted unexpectedly for operation: {operation_name}")
