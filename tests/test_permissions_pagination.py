from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

from rag_timescale.api.app import create_app


def encode_cursor(granted_at: datetime, id: UUID) -> str:
    payload = [granted_at.isoformat(), str(id)]
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


def make_key_row(key_id: UUID, granted_at: datetime, name: str = "Test Key") -> MagicMock:
    return MagicMock(
        __getitem__=MagicMock(
            side_effect=lambda k: {
                "id": key_id,
                "name": name,
                "access_level": "read",
                "granted_at": granted_at,
            }.get(k)
        )
    )


@pytest.fixture
def admin_key_id():
    return "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def collection_id():
    return "22222222-2222-2222-2222-222222222222"


@pytest.fixture
def auth_headers(admin_key_id):
    return {"X-API-Key": f"test_key_{admin_key_id}"}


class TestPermissionsPagination:
    @pytest.mark.asyncio
    async def test_list_keys_empty(self, admin_key_id, collection_id, auth_headers):
        pool, conn = make_mock_pool()
        with patch("rag_timescale.api.permissions.get_pool", return_value=pool):
            with patch("rag_timescale.api.deps.authenticate_key") as mock_auth:
                mock_auth.return_value = {"id": UUID(admin_key_id), "name": "Test", "is_active": True}
                with patch("rag_timescale.auth.keys.get_pool", return_value=pool):
                    with patch("rag_timescale.api.deps.check_collection_access", new_callable=AsyncMock) as mock_check:
                        mock_check.return_value = True
                        app = create_app()
                        transport = ASGITransport(app=app)
                        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            response = await client.get(
                                f"/api/v1/collections/{collection_id}/keys", headers=auth_headers
                            )
                            assert response.status_code == 200
                            data = response.json()
                            assert "data" in data
                            assert "next_cursor" in data
                            assert data["data"] == []
                            assert data["next_cursor"] is None

    @pytest.mark.asyncio
    async def test_list_keys_with_limit(self, admin_key_id, collection_id, auth_headers):
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = [make_key_row(UUID(int=i), now, f"Key {i}") for i in range(15)]
        pool, conn = make_mock_pool(rows)
        with patch("rag_timescale.api.permissions.get_pool", return_value=pool):
            with patch("rag_timescale.api.deps.authenticate_key") as mock_auth:
                mock_auth.return_value = {"id": UUID(admin_key_id), "name": "Test", "is_active": True}
                with patch("rag_timescale.auth.keys.get_pool", return_value=pool):
                    with patch("rag_timescale.api.deps.check_collection_access", new_callable=AsyncMock) as mock_check:
                        mock_check.return_value = True
                        app = create_app()
                        transport = ASGITransport(app=app)
                        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            response = await client.get(
                                f"/api/v1/collections/{collection_id}/keys?limit=5", headers=auth_headers
                            )
                            assert response.status_code == 200
                            data = response.json()
                            assert len(data["data"]) == 5
                            assert data["next_cursor"] is not None

    @pytest.mark.asyncio
    async def test_list_keys_with_cursor(self, admin_key_id, collection_id, auth_headers):
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        cursor_key_id = UUID("33333333-3333-3333-3333-333333333333")
        rows = [make_key_row(UUID(int=i), now, f"Key {i}") for i in range(5)]
        pool, conn = make_mock_pool(rows)
        cursor = encode_cursor(now, cursor_key_id)
        with patch("rag_timescale.api.permissions.get_pool", return_value=pool):
            with patch("rag_timescale.api.deps.authenticate_key") as mock_auth:
                mock_auth.return_value = {"id": UUID(admin_key_id), "name": "Test", "is_active": True}
                with patch("rag_timescale.auth.keys.get_pool", return_value=pool):
                    with patch("rag_timescale.api.deps.check_collection_access", new_callable=AsyncMock) as mock_check:
                        mock_check.return_value = True
                        app = create_app()
                        transport = ASGITransport(app=app)
                        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            response = await client.get(
                                f"/api/v1/collections/{collection_id}/keys?cursor={cursor}", headers=auth_headers
                            )
                            assert response.status_code == 200
                            data = response.json()
                            assert len(data["data"]) == 5
                            conn.fetch.assert_called_once()
                            call_args = conn.fetch.call_args[0][0]
                            assert "(ca.granted_at, ak.id) <" in call_args

    @pytest.mark.asyncio
    async def test_list_keys_invalid_cursor(self):
        from rag_timescale.models import decode_cursor

        with pytest.raises(ValueError):
            decode_cursor("invalid")

    @pytest.mark.asyncio
    async def test_list_keys_limit_validation(self, admin_key_id, collection_id, auth_headers):
        pool, conn = make_mock_pool()
        with patch("rag_timescale.api.permissions.get_pool", return_value=pool):
            with patch("rag_timescale.api.deps.authenticate_key") as mock_auth:
                mock_auth.return_value = {"id": UUID(admin_key_id), "name": "Test", "is_active": True}
                with patch("rag_timescale.auth.keys.get_pool", return_value=pool):
                    with patch("rag_timescale.api.deps.check_collection_access", new_callable=AsyncMock) as mock_check:
                        mock_check.return_value = True
                        app = create_app()
                        transport = ASGITransport(app=app)
                        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            response = await client.get(
                                f"/api/v1/collections/{collection_id}/keys?limit=0", headers=auth_headers
                            )
                            assert response.status_code == 422

                            response = await client.get(
                                f"/api/v1/collections/{collection_id}/keys?limit=101", headers=auth_headers
                            )
                            assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_list_keys_unauthorized(self, collection_id):
        pool, conn = make_mock_pool()
        with patch("rag_timescale.api.permissions.get_pool", return_value=pool):
            app = create_app()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(f"/api/v1/collections/{collection_id}/keys")
                assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_list_keys_forbidden(self, admin_key_id, collection_id, auth_headers):
        pool, conn = make_mock_pool()
        with patch("rag_timescale.api.permissions.get_pool", return_value=pool):
            with patch("rag_timescale.api.deps.authenticate_key") as mock_auth:
                mock_auth.return_value = {"id": UUID(admin_key_id), "name": "Test", "is_active": True}
                with patch("rag_timescale.auth.keys.get_pool", return_value=pool):
                    with patch("rag_timescale.api.deps.check_collection_access", new_callable=AsyncMock) as mock_check:
                        mock_check.return_value = False
                        app = create_app()
                        transport = ASGITransport(app=app)
                        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            response = await client.get(
                                f"/api/v1/collections/{collection_id}/keys", headers=auth_headers
                            )
                            assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_list_keys_response_structure(self, admin_key_id, collection_id, auth_headers):
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        key_id = UUID("44444444-4444-4444-4444-444444444444")
        rows = [make_key_row(key_id, now, "My API Key")]
        pool, conn = make_mock_pool(rows)
        with patch("rag_timescale.api.permissions.get_pool", return_value=pool):
            with patch("rag_timescale.api.deps.authenticate_key") as mock_auth:
                mock_auth.return_value = {"id": UUID(admin_key_id), "name": "Test", "is_active": True}
                with patch("rag_timescale.auth.keys.get_pool", return_value=pool):
                    with patch("rag_timescale.api.deps.check_collection_access", new_callable=AsyncMock) as mock_check:
                        mock_check.return_value = True
                        app = create_app()
                        transport = ASGITransport(app=app)
                        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            response = await client.get(
                                f"/api/v1/collections/{collection_id}/keys", headers=auth_headers
                            )
                            assert response.status_code == 200
                            data = response.json()
                            assert len(data["data"]) == 1
                            key_data = data["data"][0]
                            assert "key_id" in key_data
                            assert "name" in key_data
                            assert "access_level" in key_data
                            assert "granted_at" in key_data
                            assert key_data["name"] == "My API Key"
                            assert key_data["access_level"] == "read"
