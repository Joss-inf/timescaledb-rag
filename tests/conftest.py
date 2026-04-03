from __future__ import annotations

import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from rag_timescale.api.app import create_app


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def mock_pool():
    pool = MagicMock()
    conn = MagicMock()

    async def fetch_stub(query, *args):
        return []

    async def fetchrow_stub(query, *args):
        return None

    async def execute_stub(query, *args):
        return "OK"

    async def fetchval_stub(query, *args):
        return None

    conn.fetch = AsyncMock(side_effect=fetch_stub)
    conn.fetchrow = AsyncMock(side_effect=fetchrow_stub)
    conn.execute = AsyncMock(side_effect=execute_stub)
    conn.fetchval = AsyncMock(side_effect=fetchval_stub)

    pool.acquire = MagicMock(return_value=MagicMock(__aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock()))

    return pool


@pytest_asyncio.fixture(scope="function")
async def app(mock_pool):
    with patch("rag_timescale.db.connection.get_pool", return_value=mock_pool):
        with patch("rag_timescale.api.deps.authenticate_key") as mock_auth:
            mock_auth.return_value = {
                "id": "00000000-0000-0000-0000-000000000001",
                "name": "Test Key",
                "is_active": True,
            }
            application = create_app()
            yield application


@pytest_asyncio.fixture
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def admin_key_id() -> str:
    return "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def user_key_id() -> str:
    return "00000000-0000-0000-0000-000000000002"


@pytest.fixture
def auth_headers(admin_key_id):
    return {"X-API-Key": f"test_key_{admin_key_id}"}
