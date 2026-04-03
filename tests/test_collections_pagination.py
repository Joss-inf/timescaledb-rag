from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

from rag_timescale.api.app import create_app


def encode_cursor(created_at: datetime, id: UUID) -> str:
    payload = [created_at.isoformat(), str(id)]
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()


def make_mock_pool(rows=None):
    pool = MagicMock()
    conn = MagicMock()
    if rows is None:
        rows = []
    conn.fetch = AsyncMock(return_value=rows)
    conn.fetchrow = AsyncMock(return_value=None)
    conn.execute = AsyncMock(return_value="OK")
    conn.fetchval = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=MagicMock(__aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock()))
    return pool, conn


def make_row(data: dict) -> MagicMock:
    row = MagicMock()
    row.__getitem__ = MagicMock(side_effect=lambda k: data.get(k))
    return row


@pytest.fixture
def admin_key_id():
    return "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def auth_headers(admin_key_id):
    return {"X-API-Key": f"test_key_{admin_key_id}"}


@pytest.fixture
def collection_id():
    return "22222222-2222-2222-2222-222222222222"


class TestCollectionsPagination:
    @pytest.mark.asyncio
    async def test_list_collections_empty(self, admin_key_id, auth_headers):
        pool, conn = make_mock_pool()
        with patch("rag_timescale.api.collections.get_pool", return_value=pool):
            with patch("rag_timescale.api.deps.authenticate_key") as mock_auth:
                mock_auth.return_value = {"id": UUID(admin_key_id), "name": "Test", "is_active": True}
                with patch("rag_timescale.auth.keys.get_pool", return_value=pool):
                    with patch("rag_timescale.auth.keys.check_collection_access", new_callable=AsyncMock) as mock_check:
                        mock_check.return_value = True
                        app = create_app()
                        transport = ASGITransport(app=app)
                        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            response = await client.get("/api/v1/collections", headers=auth_headers)
                            assert response.status_code == 200
                            data = response.json()
                            assert "data" in data
                            assert "next_cursor" in data
                            assert data["data"] == []
                            assert data["next_cursor"] is None

    @pytest.mark.asyncio
    async def test_list_collections_with_limit(self, admin_key_id, auth_headers):
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = [make_row({"id": str(UUID(int=i)), "name": f"Col {i}", "created_at": now}) for i in range(25)]
        pool, conn = make_mock_pool(rows)
        with patch("rag_timescale.api.collections.get_pool", return_value=pool):
            with patch("rag_timescale.api.deps.authenticate_key") as mock_auth:
                mock_auth.return_value = {"id": UUID(admin_key_id), "name": "Test", "is_active": True}
                with patch("rag_timescale.auth.keys.get_pool", return_value=pool):
                    with patch("rag_timescale.auth.keys.check_collection_access", new_callable=AsyncMock) as mock_check:
                        mock_check.return_value = True
                        app = create_app()
                        transport = ASGITransport(app=app)
                        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            response = await client.get("/api/v1/collections?limit=10", headers=auth_headers)
                            assert response.status_code == 200
                            data = response.json()
                            assert len(data["data"]) == 10
                            assert data["next_cursor"] is not None
                            assert isinstance(data["next_cursor"], str)

    @pytest.mark.asyncio
    async def test_list_collections_with_cursor(self, admin_key_id, auth_headers):
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        cursor_id = UUID("11111111-1111-1111-1111-111111111111")
        rows = [make_row({"id": str(cursor_id), "name": f"Col {i}", "created_at": now}) for i in range(5)]
        pool, conn = make_mock_pool(rows)
        cursor = encode_cursor(now, cursor_id)
        with patch("rag_timescale.api.collections.get_pool", return_value=pool):
            with patch("rag_timescale.api.deps.authenticate_key") as mock_auth:
                mock_auth.return_value = {"id": UUID(admin_key_id), "name": "Test", "is_active": True}
                with patch("rag_timescale.auth.keys.get_pool", return_value=pool):
                    with patch("rag_timescale.auth.keys.check_collection_access", new_callable=AsyncMock) as mock_check:
                        mock_check.return_value = True
                        app = create_app()
                        transport = ASGITransport(app=app)
                        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            response = await client.get(f"/api/v1/collections?cursor={cursor}", headers=auth_headers)
                            assert response.status_code == 200
                            data = response.json()
                            assert len(data["data"]) == 5
                            conn.fetch.assert_called_once()
                            call_args = conn.fetch.call_args[0][0]
                            assert "(c.created_at, c.id) <" in call_args

    @pytest.mark.asyncio
    async def test_list_collections_invalid_cursor(self, admin_key_id, auth_headers):
        from rag_timescale.models import decode_cursor

        with pytest.raises(ValueError):
            decode_cursor("invalid")

    @pytest.mark.asyncio
    async def test_list_collections_limit_validation(self, admin_key_id, auth_headers):
        pool, conn = make_mock_pool()
        with patch("rag_timescale.api.collections.get_pool", return_value=pool):
            with patch("rag_timescale.api.deps.authenticate_key") as mock_auth:
                mock_auth.return_value = {"id": UUID(admin_key_id), "name": "Test", "is_active": True}
                with patch("rag_timescale.auth.keys.get_pool", return_value=pool):
                    with patch("rag_timescale.auth.keys.check_collection_access", new_callable=AsyncMock) as mock_check:
                        mock_check.return_value = True
                        app = create_app()
                        transport = ASGITransport(app=app)
                        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            response = await client.get("/api/v1/collections?limit=0", headers=auth_headers)
                            assert response.status_code == 422

                            response = await client.get("/api/v1/collections?limit=101", headers=auth_headers)
                            assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_list_collections_unauthorized(self):
        pool, conn = make_mock_pool()
        with patch("rag_timescale.api.collections.get_pool", return_value=pool):
            app = create_app()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/api/v1/collections")
                assert response.status_code == 401
