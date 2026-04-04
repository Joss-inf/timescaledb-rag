from __future__ import annotations

import uuid
from typing import Any, Optional
from structlog import get_logger

from rag_timescale.db.connection import get_pool
from rag_timescale.reranker.cross_encoder import rerank

log = get_logger()


async def hybrid_search(
    collection_id: uuid.UUID,
    query: str,
    top_k: int = 10,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
    rrf_k: int = 60,
    filters: Optional[dict] = None,
) -> tuple[list, int]:
    from rag_timescale.embeddings.provider import generate_embeddings

    filters = filters or {}
    query_embedding = (await generate_embeddings([query]))[0]

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT * FROM hybrid_search(
                p_collection_id => $1,
                p_query_text => $2,
                p_query_embedding => $3,
                p_bm25_weight => $4,
                p_vector_weight => $5,
                p_rrf_k => $6,
                p_match_count => $7,
                p_filters => $8
            )""",
            collection_id,
            query,
            query_embedding,
            bm25_weight,
            vector_weight,
            rrf_k,
            top_k,
            filters,
        )

        total = await conn.fetchval(
            "SELECT COUNT(*) FROM chunks WHERE collection_id = $1 AND deleted_at IS NULL", collection_id
        )

    results = [
        type(
            "SearchResult",
            (),
            {
                "chunk_id": r["chunk_id"],
                "document_id": r["document_id"],
                "content": r["content"],
                "score": float(r["rrf_score"]),
                "bm25_rank": r["bm25_rank"],
                "vector_rank": r["vector_rank"],
                "section_path": list(r["section_path"]) if r["section_path"] else [],
                "metadata": dict(r["metadata"]) if r["metadata"] else {},
            },
        )
        for r in rows
    ]

    return results, total


async def hybrid_search_with_rerank(
    collection_id: uuid.UUID,
    query: str,
    top_k: int = 10,
    rerank_top_k: int = 20,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
    rrf_k: int = 60,
    filters: Optional[dict] = None,
) -> tuple[list, int]:
    results, total = await hybrid_search(
        collection_id=collection_id,
        query=query,
        top_k=rerank_top_k,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        rrf_k=rrf_k,
        filters=filters,
    )

    if results:
        scored = await rerank(query, [r.content for r in results], top_k=top_k)
        results = [results[idx] for idx, score in scored]
        for (idx, score), r in zip(scored, results):
            r.score = score

    return results, total


async def get_parent_context(chunk_id: uuid.UUID, levels: int = 1) -> list[dict[str, Any]]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM get_parent_context($1, $2)",
            chunk_id,
            levels,
        )

    return [
        {
            "chunk_id": r["chunk_id"],
            "content": r["content"],
            "section_path": list(r["section_path"]) if r["section_path"] else [],
            "chunk_level": r["chunk_level"],
        }
        for r in rows
    ]
