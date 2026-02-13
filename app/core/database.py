"""Async SQLAlchemy database setup."""

import logging

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings

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


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for dependency injection."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            logger.warning(f"Database session error: {repr(e)}, rolling back")
            try:
                await session.rollback()
            except Exception:
                logger.warning("Rollback also failed (connection likely closed)")
            raise e


@asynccontextmanager
async def get_session_context() -> AsyncGenerator[AsyncSession, None]:
    """Get database session as context manager for non-DI usage."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
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
