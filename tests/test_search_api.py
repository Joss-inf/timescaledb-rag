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


def make_search_result(chunk_id: UUID, document_id: UUID, content: str, score: float) -> MagicMock:
    return MagicMock(
        __getitem__=MagicMock(
            side_effect=lambda k: {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "content": content,
                "score": score,
                "section_path": ["Section 1"],
                "metadata": {},
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


class TestSearchAPI:
    @pytest.mark.asyncio
    async def test_search_empty_results(self, admin_key_id, collection_id, auth_headers):
        pool, conn = make_mock_pool([])
        conn.fetchrow = AsyncMock(return_value={"config": {}})

        with patch("rag_timescale.api.search.get_pool", return_value=pool):
            with patch("rag_timescale.api.deps.authenticate_key") as mock_auth:
                mock_auth.return_value = {"id": UUID(admin_key_id), "name": "Test", "is_active": True}
                with patch("rag_timescale.api.deps.check_collection_access", new_callable=AsyncMock) as mock_check:
                    mock_check.return_value = True
                    with patch("rag_timescale.retrieval.hybrid.hybrid_search", new_callable=AsyncMock) as mock_search:
                        mock_search.return_value = ([], 0)
                        app = create_app()
                        transport = ASGITransport(app=app)
                        async with AsyncClient(
                            transport=transport, base_url="http://test", follow_redirects=True
                        ) as client:
                            response = await client.post(
                                f"/api/v1/collections/{collection_id}/search",
                                headers=auth_headers,
                                json={"query": "test query"},
                            )
                            assert response.status_code == 200
                            data = response.json()
                            assert "results" in data
                            assert data["results"] == []

    @pytest.mark.asyncio
    async def test_search_with_results(self, admin_key_id, collection_id, auth_headers):
        chunk_id = UUID("33333333-3333-3333-3333-333333333333")
        doc_id = UUID("44444444-4444-4444-4444-444444444444")

        pool, conn = make_mock_pool([])
        conn.fetchrow = AsyncMock(return_value={"config": {}})

        mock_result = MagicMock()
        mock_result.chunk_id = chunk_id
        mock_result.document_id = doc_id
        mock_result.content = "Test content about machine learning"
        mock_result.score = 0.95
        mock_result.section_path = ["Chapter 1"]
        mock_result.metadata = {}

        with patch("rag_timescale.api.search.get_pool", return_value=pool):
            with patch("rag_timescale.api.deps.authenticate_key") as mock_auth:
                mock_auth.return_value = {"id": UUID(admin_key_id), "name": "Test", "is_active": True}
                with patch("rag_timescale.api.deps.check_collection_access", new_callable=AsyncMock) as mock_check:
                    mock_check.return_value = True
                    with patch("rag_timescale.retrieval.hybrid.hybrid_search", new_callable=AsyncMock) as mock_search:
                        mock_search.return_value = ([mock_result], 100)
                        app = create_app()
                        transport = ASGITransport(app=app)
                        async with AsyncClient(
                            transport=transport, base_url="http://test", follow_redirects=True
                        ) as client:
                            response = await client.post(
                                f"/api/v1/collections/{collection_id}/search",
                                headers=auth_headers,
                                json={"query": "machine learning", "top_k": 10},
                            )
                            assert response.status_code == 200
                            data = response.json()
                            assert len(data["results"]) == 1
                            assert data["query"] == "machine learning"
                            assert "total_chunks_searched" in data
                            assert "elapsed_ms" in data

    @pytest.mark.asyncio
    async def test_search_with_rerank(self, admin_key_id, collection_id, auth_headers):
        pool, conn = make_mock_pool([])
        conn.fetchrow = AsyncMock(return_value={"config": {}})

        mock_result = MagicMock()
        mock_result.chunk_id = UUID("33333333-3333-3333-3333-333333333333")
        mock_result.document_id = UUID("44444444-4444-4444-4444-444444444444")
        mock_result.content = "Test content"
        mock_result.score = 0.9
        mock_result.section_path = []
        mock_result.metadata = {}

        with patch("rag_timescale.api.search.get_pool", return_value=pool):
            with patch("rag_timescale.api.deps.authenticate_key") as mock_auth:
                mock_auth.return_value = {"id": UUID(admin_key_id), "name": "Test", "is_active": True}
                with patch("rag_timescale.api.deps.check_collection_access", new_callable=AsyncMock) as mock_check:
                    mock_check.return_value = True
                    with patch(
                        "rag_timescale.retrieval.hybrid.hybrid_search_with_rerank", new_callable=AsyncMock
                    ) as mock_search:
                        mock_search.return_value = ([mock_result], 50)
                        app = create_app()
                        transport = ASGITransport(app=app)
                        async with AsyncClient(
                            transport=transport, base_url="http://test", follow_redirects=True
                        ) as client:
                            response = await client.post(
                                f"/api/v1/collections/{collection_id}/search",
                                headers=auth_headers,
                                json={"query": "test", "rerank": True, "rerank_top_k": 5},
                            )
                            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_search_bm25_only(self, admin_key_id, collection_id, auth_headers):
        pool, conn = make_mock_pool([])
        conn.fetchrow = AsyncMock(return_value={"config": {}})

        with patch("rag_timescale.api.search.get_pool", return_value=pool):
            with patch("rag_timescale.api.deps.authenticate_key") as mock_auth:
                mock_auth.return_value = {"id": UUID(admin_key_id), "name": "Test", "is_active": True}
                with patch("rag_timescale.api.deps.check_collection_access", new_callable=AsyncMock) as mock_check:
                    mock_check.return_value = True
                    with patch("rag_timescale.retrieval.hybrid.hybrid_search", new_callable=AsyncMock) as mock_search:
                        mock_search.return_value = ([], 0)
                        app = create_app()
                        transport = ASGITransport(app=app)
                        async with AsyncClient(
                            transport=transport, base_url="http://test", follow_redirects=True
                        ) as client:
                            response = await client.post(
                                f"/api/v1/collections/{collection_id}/search/bm25",
                                headers=auth_headers,
                                json={"query": "keywords", "top_k": 10},
                            )
                            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_search_vector_only(self, admin_key_id, collection_id, auth_headers):
        pool, conn = make_mock_pool([])
        conn.fetchrow = AsyncMock(return_value={"config": {}})

        with patch("rag_timescale.api.search.get_pool", return_value=pool):
            with patch("rag_timescale.api.deps.authenticate_key") as mock_auth:
                mock_auth.return_value = {"id": UUID(admin_key_id), "name": "Test", "is_active": True}
                with patch("rag_timescale.api.deps.check_collection_access", new_callable=AsyncMock) as mock_check:
                    mock_check.return_value = True
                    with patch("rag_timescale.retrieval.hybrid.hybrid_search", new_callable=AsyncMock) as mock_search:
                        mock_search.return_value = ([], 0)
                        app = create_app()
                        transport = ASGITransport(app=app)
                        async with AsyncClient(
                            transport=transport, base_url="http://test", follow_redirects=True
                        ) as client:
                            response = await client.post(
                                f"/api/v1/collections/{collection_id}/search/vector",
                                headers=auth_headers,
                                json={"query": "semantic search", "top_k": 10},
                            )
                            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_search_raw_endpoint(self, admin_key_id, collection_id, auth_headers):
        pool, conn = make_mock_pool([])
        conn.fetchrow = AsyncMock(return_value={"config": {}})

        with patch("rag_timescale.api.search.get_pool", return_value=pool):
            with patch("rag_timescale.api.deps.authenticate_key") as mock_auth:
                mock_auth.return_value = {"id": UUID(admin_key_id), "name": "Test", "is_active": True}
                with patch("rag_timescale.api.deps.check_collection_access", new_callable=AsyncMock) as mock_check:
                    mock_check.return_value = True
                    with patch("rag_timescale.retrieval.hybrid.hybrid_search", new_callable=AsyncMock) as mock_search:
                        mock_search.return_value = ([], 0)
                        app = create_app()
                        transport = ASGITransport(app=app)
                        async with AsyncClient(
                            transport=transport, base_url="http://test", follow_redirects=True
                        ) as client:
                            response = await client.post(
                                f"/api/v1/collections/{collection_id}/search/raw",
                                headers=auth_headers,
                                json={"query": "fast search"},
                            )
                            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_search_unauthorized(self, collection_id):
        pool, conn = make_mock_pool()
        with patch("rag_timescale.api.search.get_pool", return_value=pool):
            app = create_app()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(f"/api/v1/collections/{collection_id}/search", json={"query": "test"})
                assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_search_forbidden(self, admin_key_id, collection_id, auth_headers):
        pool, conn = make_mock_pool()
        with patch("rag_timescale.api.search.get_pool", return_value=pool):
            with patch("rag_timescale.api.deps.authenticate_key") as mock_auth:
                mock_auth.return_value = {"id": UUID(admin_key_id), "name": "Test", "is_active": True}
                with patch("rag_timescale.api.deps.check_collection_access", new_callable=AsyncMock) as mock_check:
                    mock_check.return_value = False
                    app = create_app()
                    transport = ASGITransport(app=app)
                    async with AsyncClient(transport=transport, base_url="http://test") as client:
                        response = await client.post(
                            f"/api/v1/collections/{collection_id}/search", headers=auth_headers, json={"query": "test"}
                        )
                        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_search_with_filters(self, admin_key_id, collection_id, auth_headers):
        pool, conn = make_mock_pool([])
        conn.fetchrow = AsyncMock(return_value={"config": {}})

        with patch("rag_timescale.api.search.get_pool", return_value=pool):
            with patch("rag_timescale.api.deps.authenticate_key") as mock_auth:
                mock_auth.return_value = {"id": UUID(admin_key_id), "name": "Test", "is_active": True}
                with patch("rag_timescale.api.deps.check_collection_access", new_callable=AsyncMock) as mock_check:
                    mock_check.return_value = True
                    with patch("rag_timescale.retrieval.hybrid.hybrid_search", new_callable=AsyncMock) as mock_search:
                        mock_search.return_value = ([], 0)
                        app = create_app()
                        transport = ASGITransport(app=app)
                        async with AsyncClient(
                            transport=transport, base_url="http://test", follow_redirects=True
                        ) as client:
                            response = await client.post(
                                f"/api/v1/collections/{collection_id}/search",
                                headers=auth_headers,
                                json={"query": "test", "filters": {"chunk_level": 1}},
                            )
                            assert response.status_code == 200
                            mock_search.assert_called_once()
                            call_kwargs = mock_search.call_args[1]
                            assert call_kwargs["filters"] == {"chunk_level": 1}

    @pytest.mark.asyncio
    async def test_search_with_custom_weights(self, admin_key_id, collection_id, auth_headers):
        pool, conn = make_mock_pool([])
        conn.fetchrow = AsyncMock(return_value={"config": {}})

        with patch("rag_timescale.api.search.get_pool", return_value=pool):
            with patch("rag_timescale.api.deps.authenticate_key") as mock_auth:
                mock_auth.return_value = {"id": UUID(admin_key_id), "name": "Test", "is_active": True}
                with patch("rag_timescale.api.deps.check_collection_access", new_callable=AsyncMock) as mock_check:
                    mock_check.return_value = True
                    with patch("rag_timescale.retrieval.hybrid.hybrid_search", new_callable=AsyncMock) as mock_search:
                        mock_search.return_value = ([], 0)
                        app = create_app()
                        transport = ASGITransport(app=app)
                        async with AsyncClient(
                            transport=transport, base_url="http://test", follow_redirects=True
                        ) as client:
                            response = await client.post(
                                f"/api/v1/collections/{collection_id}/search",
                                headers=auth_headers,
                                json={"query": "test", "bm25_weight": 0.3, "vector_weight": 0.7},
                            )
                            assert response.status_code == 200
