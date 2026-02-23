"""Database kernel utilities for short-lived read/write operations."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from time import monotonic
from typing import TypeVar

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session_context
from app.core.db_retry import is_transient_connection_error

logger = logging.getLogger(__name__)

_ResultT = TypeVar("_ResultT")


class DbKernelError(RuntimeError):
    """Base error for DB kernel operations."""


class TransientDbError(DbKernelError):
    """Transient DB failure that can usually be retried."""


class ConflictError(DbKernelError):
    """Write conflict (usually integrity/unique constraint)."""


class PermanentDbError(DbKernelError):
    """Non-transient DB failure."""


def _translate_error(exc: Exception) -> DbKernelError:
    if isinstance(exc, IntegrityError):
        return ConflictError(str(exc))
    if is_transient_connection_error(exc):
        return TransientDbError(str(exc))
    return PermanentDbError(str(exc))


async def db_read(
    fn: Callable[[AsyncSession], Awaitable[_ResultT]],
    *,
    operation_name: str,
) -> _ResultT:
    """Execute a read operation in a short-lived session."""
    started = monotonic()
    try:
        async with get_session_context(commit_on_exit=False) as session:
            result = await fn(session)
        logger.debug(
            "DB read operation completed",
            extra={
                "operation": operation_name,
                "duration_ms": round((monotonic() - started) * 1000, 2),
            },
        )
        return result
    except Exception as exc:
        translated = _translate_error(exc)
        logger.warning(
            "DB read operation failed",
            extra={
                "operation": operation_name,
                "duration_ms": round((monotonic() - started) * 1000, 2),
                "failure_class": type(translated).__name__,
            },
        )
        raise translated from exc


async def db_write_no_retry(
    fn: Callable[[AsyncSession], Awaitable[_ResultT]],
    *,
    operation_name: str,
) -> _ResultT:
    """Execute a write operation in a short-lived session without retries."""
    started = monotonic()
    try:
        async with get_session_context(commit_on_exit=False) as session:
            result = await fn(session)
            await session.commit()
        logger.debug(
            "DB write operation completed",
            extra={
                "operation": operation_name,
                "duration_ms": round((monotonic() - started) * 1000, 2),
                "attempt": 1,
                "max_attempts": 1,
            },
        )
        return result
    except Exception as exc:
        translated = _translate_error(exc)
        logger.warning(
            "DB write operation failed",
            extra={
                "operation": operation_name,
                "duration_ms": round((monotonic() - started) * 1000, 2),
                "failure_class": type(translated).__name__,
                "attempt": 1,
                "max_attempts": 1,
            },
        )
        raise translated from exc


async def db_write(
    fn: Callable[[AsyncSession], Awaitable[_ResultT]],
    *,
    operation_name: str,
    attempts: int = 3,
    base_delay_seconds: float = 0.2,
) -> _ResultT:
    """Execute a write operation with retries for transient failures."""
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    started = monotonic()
    for attempt in range(1, attempts + 1):
        try:
            async with get_session_context(commit_on_exit=False) as session:
                result = await fn(session)
                await session.commit()
            logger.debug(
                "DB write operation completed",
                extra={
                    "operation": operation_name,
                    "duration_ms": round((monotonic() - started) * 1000, 2),
                    "attempt": attempt,
                    "max_attempts": attempts,
                },
            )
            return result
        except Exception as exc:
            translated = _translate_error(exc)
            is_retryable = isinstance(translated, TransientDbError) and attempt < attempts
            logger.warning(
                "DB write operation failed",
                extra={
                    "operation": operation_name,
                    "duration_ms": round((monotonic() - started) * 1000, 2),
                    "failure_class": type(translated).__name__,
                    "attempt": attempt,
                    "max_attempts": attempts,
                    "will_retry": is_retryable,
                },
            )
            if not is_retryable:
                raise translated from exc
            await asyncio.sleep(base_delay_seconds * attempt)

    raise RuntimeError(f"DB write retry loop exhausted unexpectedly: {operation_name}")
