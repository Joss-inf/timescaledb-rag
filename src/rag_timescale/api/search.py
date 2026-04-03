from __future__ import annotations

import time

from fastapi import APIRouter

from rag_timescale.api.deps import RequireKey
from rag_timescale.db.connection import get_pool
from rag_timescale.models import SearchRequest, SearchResponse
from rag_timescale.retrieval.hybrid import get_parent_context, hybrid_search, hybrid_search_with_rerank

router = APIRouter(prefix="/api/v1/collections", tags=["search"])


@router.post("/{collection_id}/search", response_model=SearchResponse)
async def search(collection_id, body: SearchRequest, key_info: RequireKey):
    start = time.time()
    config = await _get_config(collection_id)
    bm25_w = body.bm25_weight or config.get("bm25_weight", 0.5)
    vec_w = body.vector_weight or config.get("vector_weight", 0.5)
    rrf_k = config.get("rrf_k", 60)

    if body.rerank:
        results, searched = await hybrid_search_with_rerank(
            collection_id, body.query, body.top_k, body.rerank_top_k, bm25_w, vec_w, rrf_k, body.filters
        )
    else:
        results, searched = await hybrid_search(
            collection_id, body.query, body.top_k, bm25_w, vec_w, rrf_k, body.filters
        )

    if body.include_parent_context:
        for r in results:
            parents = await get_parent_context(r.chunk_id, levels=body.parent_context_levels)
            if parents:
                r.metadata["parent_context"] = parents

    return SearchResponse(
        query=body.query,
        results=results,
        total_chunks_searched=searched,
        elapsed_ms=round((time.time() - start) * 1000, 2),
    )


@router.post("/{collection_id}/search/raw", response_model=SearchResponse)
async def search_raw(collection_id, body: SearchRequest, key_info: RequireKey):
    start = time.time()
    config = await _get_config(collection_id)
    bm25_w = body.bm25_weight or config.get("bm25_weight", 0.5)
    vec_w = body.vector_weight or config.get("vector_weight", 0.5)
    rrf_k = config.get("rrf_k", 60)
    results, searched = await hybrid_search(collection_id, body.query, body.top_k, bm25_w, vec_w, rrf_k, body.filters)
    return SearchResponse(
        query=body.query,
        results=results,
        total_chunks_searched=searched,
        elapsed_ms=round((time.time() - start) * 1000, 2),
    )


@router.post("/{collection_id}/search/similar", response_model=SearchResponse)
async def search_similar(collection_id, body: SearchRequest, key_info: RequireKey):
    start = time.time()
    config = await _get_config(collection_id)
    bm25_w = body.bm25_weight or config.get("bm25_weight", 0.5)
    vec_w = body.vector_weight or config.get("vector_weight", 0.5)
    rrf_k = config.get("rrf_k", 60)
    results, searched = await hybrid_search(collection_id, body.query, body.top_k, bm25_w, vec_w, rrf_k, body.filters)
    for r in results:
        parents = await get_parent_context(r.chunk_id, levels=body.parent_context_levels)
        if parents:
            r.metadata["parent_context"] = parents
    return SearchResponse(
        query=body.query,
        results=results,
        total_chunks_searched=searched,
        elapsed_ms=round((time.time() - start) * 1000, 2),
    )


async def _get_config(collection_id) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT config FROM collections WHERE id = $1", collection_id)
    return row["config"] if row and row["config"] else {}
