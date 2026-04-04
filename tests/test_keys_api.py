from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

from rag_timescale.api.app import create_app


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


@pytest.fixture
def admin_key_id():
    return "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def user_key_id():
    return "00000000-0000-0000-0000-000000000002"


@pytest.fixture
def auth_headers(admin_key_id):
    return {"X-API-Key": f"test_key_{admin_key_id}"}


class TestKeysAPI:
    @pytest.mark.asyncio
    async def test_create_key(self, admin_key_id, auth_headers):
        pool, conn = make_mock_pool()

        key_id = UUID("550e8400-e29b-41d4-a716-446655440000")
        now = datetime.now(timezone.utc)

        with patch("rag_timescale.api.keys.get_pool", return_value=pool):
            with patch("rag_timescale.auth.keys.create_api_key", new_callable=AsyncMock) as mock_create:
                mock_create.return_value = (
                    "rag_testkey123456789",
                    {
                        "id": key_id,
                        "name": "Test Key",
                        "created_at": now,
                        "expires_at": None,
                    },
                )
                app = create_app()
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.post("/api/v1/keys", json={"name": "Test Key"})
                    assert response.status_code == 201
                    data = response.json()
                    assert "key" in data
                    assert data["name"] == "Test Key"
                    assert data["id"] == str(key_id)

    @pytest.mark.asyncio
    async def test_create_key_with_permissions(self, admin_key_id, auth_headers):
        pool, conn = make_mock_pool()

        with patch("rag_timescale.api.keys.get_pool", return_value=pool):
            with patch("rag_timescale.auth.keys.create_api_key", new_callable=AsyncMock) as mock_create:
                mock_create.return_value = (
                    "rag_testkey123",
                    {
                        "id": UUID(),
                        "name": "Admin Key",
                        "created_at": datetime.now(timezone.utc),
                        "expires_at": None,
                    },
                )
                app = create_app()
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.post(
                        "/api/v1/keys",
                        json={
                            "name": "Admin Key",
                            "permissions": {"collections": ["*"], "roles": ["read", "write", "admin"]},
                        },
                    )
                    assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_delete_own_key(self, admin_key_id, auth_headers):
        pool, conn = make_mock_pool()

        key_id = UUID(admin_key_id)

        with patch("rag_timescale.api.keys.get_pool", return_value=pool):
            with patch("rag_timescale.api.deps.authenticate_key") as mock_auth:
                mock_auth.return_value = {"id": key_id, "name": "Test", "is_active": True}
                with patch("rag_timescale.auth.keys.revoke_api_key", new_callable=AsyncMock) as mock_revoke:
                    mock_revoke.return_value = True
                    app = create_app()
                    transport = ASGITransport(app=app)
                    async with AsyncClient(transport=transport, base_url="http://test") as client:
                        response = await client.delete(f"/api/v1/keys/{admin_key_id}", headers=auth_headers)
                        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_delete_other_key_forbidden(self, admin_key_id, user_key_id, auth_headers):
        pool, conn = make_mock_pool()

        with patch("rag_timescale.api.keys.get_pool", return_value=pool):
            with patch("rag_timescale.api.deps.authenticate_key") as mock_auth:
                mock_auth.return_value = {"id": UUID(admin_key_id), "name": "Test", "is_active": True}
                app = create_app()
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.delete(f"/api/v1/keys/{user_key_id}", headers=auth_headers)
                    assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self, admin_key_id, auth_headers):
        pool, conn = make_mock_pool()

        nonexistent_id = "99999999-9999-9999-9999-999999999999"

        with patch("rag_timescale.api.keys.get_pool", return_value=pool):
            with patch("rag_timescale.api.deps.authenticate_key") as mock_auth:
                mock_auth.return_value = {"id": UUID(admin_key_id), "name": "Test", "is_active": True}
                with patch("rag_timescale.auth.keys.revoke_api_key", new_callable=AsyncMock) as mock_revoke:
                    mock_revoke.return_value = False
                    app = create_app()
                    transport = ASGITransport(app=app)
                    async with AsyncClient(transport=transport, base_url="http://test") as client:
                        response = await client.delete(f"/api/v1/keys/{nonexistent_id}", headers=auth_headers)
                        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_list_keys_unauthorized(self):
        pool, conn = make_mock_pool()
        with patch("rag_timescale.api.keys.get_pool", return_value=pool):
            app = create_app()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/api/v1/keys")
                assert response.status_code in [401, 404, 422]
