from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Optional
from functools import lru_cache
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, status
from structlog import get_logger

from rag_timescale.api.deps import RequireRead  # ← changer RequireKey → RequireRead
from rag_timescale.db.connection import get_pool
from rag_timescale.models import SearchRequest, SearchResponse
from rag_timescale.retrieval.hybrid import (
    get_parent_context,
    hybrid_search,
    hybrid_search_with_rerank,
)

log = get_logger()
router = APIRouter()

# Cache config collection (TTL 5 secondes)
_config_cache: dict[str, tuple[float, dict]] = {}
CACHE_TTL = 5.0


async def _get_config(collection_id: uuid.UUID) -> dict[str, Any]:
    """Récupère la config avec cache court."""
    key = str(collection_id)
    now = time.time()
    if key in _config_cache:
        ts, cfg = _config_cache[key]
        if now - ts < CACHE_TTL:
            return cfg

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT config FROM collections WHERE id = $1", collection_id)
    cfg = dict(row["config"]) if row and row["config"] else {}
    _config_cache[key] = (now, cfg)
    return cfg


def _invalidate_config_cache(collection_id: str) -> None:
    _config_cache.pop(collection_id, None)


# ----------------------------------------------------------------------
# Endpoint principal (hybride complet)
# ----------------------------------------------------------------------
@router.post("/", response_model=SearchResponse)
async def search(
    collection_id: uuid.UUID,
    body: SearchRequest,
    key_info: RequireRead,  # ← RequireRead au lieu de RequireKey
) -> SearchResponse:
    config = await _get_config(collection_id)

    bm25_weight = body.bm25_weight if body.bm25_weight is not None else config.get("bm25_weight", 0.5)
    vector_weight = body.vector_weight if body.vector_weight is not None else config.get("vector_weight", 0.5)
    rrf_k = config.get("rrf_k", 60)
    diskann_search_list = body.diskann_search_list or config.get("diskann_search_list", 200)
    diskann_rescore = body.diskann_rescore or config.get("diskann_rescore", 100)

    results, total_searched, elapsed_ms = await _execute_search(
        collection_id=collection_id,
        query=body.query,
        top_k=body.top_k,
        rerank=body.rerank,
        rerank_top_k=body.rerank_top_k,
        include_parent_context=body.include_parent_context,
        parent_context_levels=body.parent_context_levels,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        rrf_k=rrf_k,
        filters=body.filters,
        diskann_search_list=diskann_search_list,
        diskann_rescore=diskann_rescore,
    )

    log.info("search_completed",
             collection_id=str(collection_id),
             query=body.query[:50],
             top_k=body.top_k,
             rerank=body.rerank,
             results_count=len(results),
             total_searched=total_searched,
             elapsed_ms=round(elapsed_ms, 2))
    return SearchResponse(
        query=body.query,
        results=results,
        total_chunks_searched=total_searched,
        elapsed_ms=round(elapsed_ms, 2)
    )


# ----------------------------------------------------------------------
# Endpoint BM25 only
# ----------------------------------------------------------------------
@router.post("/bm25", response_model=SearchResponse)
async def search_bm25(
    collection_id: uuid.UUID,
    body: SearchRequest,
    key_info: RequireRead,
) -> SearchResponse:
    """Recherche BM25 uniquement (pas de vecteur)."""
    config = await _get_config(collection_id)
    bm25_weight = config.get("bm25_weight", 0.5)
    # Force vector_weight = 0
    results, total_searched, elapsed_ms = await _execute_search(
        collection_id=collection_id,
        query=body.query,
        top_k=body.top_k or 10,
        bm25_weight=bm25_weight,
        vector_weight=0.0,
        rrf_k=60,
        filters=body.filters,
        rerank=False,
        include_parent_context=False,
        diskann_search_list=200,
        diskann_rescore=100,
    )
    return SearchResponse(
        query=body.query,
        results=results,
        total_chunks_searched=total_searched,
        elapsed_ms=round(elapsed_ms, 2)
    )


# ----------------------------------------------------------------------
# Endpoint Vector only
# ----------------------------------------------------------------------
@router.post("/vector", response_model=SearchResponse)
async def search_vector(
    collection_id: uuid.UUID,
    body: SearchRequest,
    key_info: RequireRead,
) -> SearchResponse:
    """Recherche vectorielle uniquement (pas de BM25)."""
    config = await _get_config(collection_id)
    vector_weight = config.get("vector_weight", 0.5)
    results, total_searched, elapsed_ms = await _execute_search(
        collection_id=collection_id,
        query=body.query,
        top_k=body.top_k or 10,
        bm25_weight=0.0,
        vector_weight=vector_weight,
        rrf_k=60,
        filters=body.filters,
        rerank=False,
        include_parent_context=False,
        diskann_search_list=config.get("diskann_search_list", 200),
        diskann_rescore=config.get("diskann_rescore", 100),
    )
    return SearchResponse(
        query=body.query,
        results=results,
        total_chunks_searched=total_searched,
        elapsed_ms=round(elapsed_ms, 2)
    )


# ----------------------------------------------------------------------
# Endpoint raw (sans rerank, sans parent)
# ----------------------------------------------------------------------
@router.post("/raw", response_model=SearchResponse)
async def search_raw(
    collection_id: uuid.UUID,
    body: SearchRequest,
    key_info: RequireRead,
) -> SearchResponse:
    """Recherche brute, rapide, sans reranking ni contexte parent."""
    config = await _get_config(collection_id)
    results, total_searched, elapsed_ms = await _execute_search(
        collection_id=collection_id,
        query=body.query,
        top_k=body.top_k or 10,
        bm25_weight=config.get("bm25_weight", 0.5),
        vector_weight=config.get("vector_weight", 0.5),
        rrf_k=config.get("rrf_k", 60),
        filters=body.filters,
        rerank=False,
        include_parent_context=False,
        diskann_search_list=config.get("diskann_search_list", 200),
        diskann_rescore=config.get("diskann_rescore", 100),
    )
    return SearchResponse(
        query=body.query,
        results=results,
        total_chunks_searched=total_searched,
        elapsed_ms=round(elapsed_ms, 2)
    )


# ----------------------------------------------------------------------
# Fonction utilitaire _execute_search (avec SET LOCAL DiskANN)
# ----------------------------------------------------------------------
async def _execute_search(
    collection_id: uuid.UUID,
    query: str,
    top_k: int,
    rerank: bool = False,
    rerank_top_k: Optional[int] = None,
    include_parent_context: bool = False,
    parent_context_levels: int = 1,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
    rrf_k: int = 60,
    filters: Optional[dict] = None,
    diskann_search_list: int = 200,
    diskann_rescore: int = 100,
) -> tuple[list, int, float]:
    filters = filters or {}
    t_start = time.perf_counter()

    pool = await get_pool()
    async with pool.acquire() as conn:
        # Ajustements locaux pour DiskANN
        await conn.execute(f"SET LOCAL diskann.query_search_list_size = {diskann_search_list}")
        await conn.execute(f"SET LOCAL diskann.query_rescore = {diskann_rescore}")

        if rerank:
            rerank_limit = rerank_top_k or max(top_k * 2, 20)
            results, total_searched = await hybrid_search_with_rerank(
                collection_id=collection_id,
                query=query,
                top_k=top_k,
                rerank_top_k=rerank_limit,
                bm25_weight=bm25_weight,
                vector_weight=vector_weight,
                rrf_k=rrf_k,
                filters=filters,
            )
        else:
            results, total_searched = await hybrid_search(
                collection_id=collection_id,
                query=query,
                top_k=top_k,
                bm25_weight=bm25_weight,
                vector_weight=vector_weight,
                rrf_k=rrf_k,
                filters=filters,
            )

    # Parent context (en dehors de la connexion, pour libérer plus tôt)
    if include_parent_context and results:
        semaphore = asyncio.Semaphore(10)
        async def fetch_one(r):
            async with semaphore:
                return await get_parent_context(r.chunk_id, parent_context_levels)
        parent_contexts = await asyncio.gather(*[fetch_one(r) for r in results])
        for r, parents in zip(results, parent_contexts):
            if parents:
                if r.metadata is None:
                    r.metadata = {}
                r.metadata["parent_context"] = parents

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    return results, total_searched, elapsed_ms