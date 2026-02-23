"""Async SQLAlchemy database setup."""

import logging

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.exc import InterfaceError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings
from app.core.db_retry import is_transient_connection_error

logger = logging.getLogger(__name__)

# Create async engine
engine = create_async_engine(
    settings.database_url,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    echo=settings.database_echo,
    pool_pre_ping=True,
    pool_recycle=300,  # Recycle connections every 5 min to avoid server-side timeouts
)

# Session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


def _has_pending_state(session: AsyncSession) -> bool:
    return bool(session.new or session.dirty or session.deleted)


async def rollback_read_only_transaction(
    session: AsyncSession,
    *,
    context: str,
) -> None:
    """Close an open read-only transaction only when no ORM changes are pending.

    This prevents long-running read-heavy flows from keeping idle transactions
    open across external I/O (LLM/API calls). We use commit (not rollback)
    because rollback expires ORM attributes and can trigger async lazy-load
    attribute access failures (`MissingGreenlet`) later in the same flow.
    """
    in_transaction = getattr(session, "in_transaction", None)
    if not callable(in_transaction):
        return
    if not in_transaction():
        return
    if _has_pending_state(session):
        return

    try:
        await session.commit()
    except Exception as exc:
        if is_transient_connection_error(exc):
            logger.debug(
                "Ignoring transient commit failure for read-only transaction",
                extra={"context": context},
            )
            return
        raise


async def _finalize_session(
    session: AsyncSession,
    *,
    commit_on_exit: bool,
    context: str,
) -> None:
    if commit_on_exit:
        await session.commit()
        return

    if _has_pending_state(session):
        raise RuntimeError(
            "Session has pending ORM changes but commit_on_exit=False. "
            "Commit explicitly or use commit_on_exit=True."
        )

    await rollback_read_only_transaction(session, context=context)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for dependency injection."""
    async with async_session_maker() as session:
        try:
            yield session
            await _finalize_session(session, commit_on_exit=True, context="get_session")
        except InterfaceError as e:
            has_active_transaction = session.in_transaction()
            has_pending_state = _has_pending_state(session)
            if not has_active_transaction and not has_pending_state:
                # Connection closed after work already committed/rolled back.
                logger.debug("Session connection already closed during cleanup, ignoring")
                return
            logger.warning(f"Database interface error with active transaction: {repr(e)}, rolling back")
            try:
                await session.rollback()
            except Exception:
                logger.warning("Rollback also failed (connection likely closed)")
            raise
        except Exception as e:
            logger.warning(f"Database session error: {repr(e)}, rolling back")
            try:
                await session.rollback()
            except Exception:
                logger.warning("Rollback also failed (connection likely closed)")
            raise e


@asynccontextmanager
async def get_session_context(
    *,
    commit_on_exit: bool = True,
) -> AsyncGenerator[AsyncSession, None]:
    """Get database session as context manager for non-DI usage."""
    async with async_session_maker() as session:
        try:
            yield session
            await _finalize_session(
                session,
                commit_on_exit=commit_on_exit,
                context="get_session_context",
            )
        except InterfaceError as e:
            has_active_transaction = session.in_transaction()
            has_pending_state = _has_pending_state(session)
            if not has_active_transaction and not has_pending_state:
                logger.debug("Session connection already closed during cleanup, ignoring")
                return
            logger.warning(f"Database interface error with active transaction: {repr(e)}, rolling back")
            try:
                await session.rollback()
            except Exception:
                logger.warning("Rollback also failed (connection likely closed)")
            raise
        except Exception as e:
            logger.warning(f"Database session error: {repr(e)}, rolling back")
            try:
                await session.rollback()
            except Exception:
                logger.warning("Rollback also failed (connection likely closed)")
            raise e


async def init_db() -> None:
    """Initialize database (create tables if needed)."""
    logger.info("Initializing database tables")
    from app.models.base import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connections."""
    logger.info("Closing database connections")
    await engine.dispose()
