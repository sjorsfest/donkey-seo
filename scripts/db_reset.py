"""Development helper to reset the Postgres public schema."""

from __future__ import annotations

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings


async def _reset_schema() -> None:
    engine = create_async_engine(settings.database_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
            await conn.execute(text("CREATE SCHEMA public"))
            await conn.execute(text("GRANT ALL ON SCHEMA public TO public"))
    finally:
        await engine.dispose()


def main() -> int:
    asyncio.run(_reset_schema())
    print("Database schema reset complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
