from __future__ import annotations

import asyncpg
from structlog import get_logger

from rag_timescale.config import settings

log = get_logger()
_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=settings.database.dsn,
            min_size=5,
            max_size=settings.database.pool_size,
            max_inactive_connection_lifetime=300,
            command_timeout=60,
        )
        log.info("database_pool_created", host=settings.database.host, port=settings.database.port)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        log.info("database_pool_closed")
