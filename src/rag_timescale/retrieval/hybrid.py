from __future__ import annotations

import uuid
from typing import Any

from structlog import get_logger

from rag_timescale.db.connection import get_pool
from rag_timescale.embeddings.provider import generate_embedding
from rag_timescale.models import SearchResult
from rag_timescale.reranker.cross_encoder import rerank

log = get_logger()


async def hybrid_search(
    collection_id: uuid.UUID,
    query: str,
    top_k: int = 10,
    bm25_limit: int = 20,
    vector_limit: int = 20,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
    rrf_k: int = 60,
    filters: dict[str, Any] | None = None,
) -> tuple[list[SearchResult], int]:
    query_embedding = generate_embedding(query)
    pool = await get_pool()
    
    # On prépare les filtres en JSON pour PostgreSQL
    # S'il n'y a pas de filtres, on envoie un objet JSON vide {}
    sql_filters = json.dumps(filters) if filters else "{}"

    async with pool.acquire() as conn:
        # On ajoute le $10 pour correspondre au paramètre p_filters de ta fonction SQL
        rows = await conn.fetch(
            """
            SELECT * FROM hybrid_search(
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
            )
            """,
            collection_id,      # $1
            query,              # $2
            query_embedding,    # $3
            bm25_limit,         # $4
            vector_limit,       # $5
            bm25_weight,        # $6
            vector_weight,      # $7
            rrf_k,              # $8
            top_k * 2,          # $9 (p_match_count)
            sql_filters         # $10 (p_filters)
        )

    results: list[SearchResult] = []
    for row in rows:
        results.append(
            SearchResult(
                chunk_id=row["id"],
                document_id=row["document_id"],
                content=row["content"],
                score=row["rrf_score"],
                bm25_rank=row["bm25_rank"],
                vector_rank=row["vector_rank"],
                section_path=list(row["section_path"]) if row["section_path"] else [],
                metadata=dict(row["metadata"]) if row["metadata"] else {},
            )
        )

    total_searched = len(results)
    log.info("hybrid_search_completed", query=query, total_results=len(results), filters=filters)
    return results, total_searched


async def hybrid_search_with_rerank(
    collection_id: uuid.UUID,
    query: str,
    top_k: int = 10,
    rerank_top_k: int = 20,
    bm25_limit: int = 20,
    vector_limit: int = 20,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
    rrf_k: int = 60,
    filters: dict[str, Any] | None = None,
) -> tuple[list[SearchResult], int]:
    results, total_searched = await hybrid_search(
        collection_id=collection_id,
        query=query,
        top_k=rerank_top_k,
        bm25_limit=bm25_limit,
        vector_limit=vector_limit,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        rrf_k=rrf_k,
        filters=filters,
    )

    if not results:
        return [], total_searched

    documents = [r.content for r in results]
    ranked_indices = rerank(query, documents, top_k=top_k)

    reranked = []
    for orig_idx, rerank_score in ranked_indices:
        result = results[orig_idx]
        result.score = rerank_score
        reranked.append(result)

    log.info("rerank_completed", query=query, before=len(results), after=len(reranked))
    return reranked, total_searched


async def get_parent_context(
    chunk_id: uuid.UUID,
    levels: int = 1,
) -> list[dict[str, Any]]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM get_parent_context($1, $2)
            """,
            chunk_id,
            levels,
        )

    return [
        {
            "chunk_id": row["chunk_id"],
            "content": row["content"],
            "section_path": list(row["section_path"]) if row["section_path"] else [],
            "chunk_level": row["chunk_level"],
        }
        for row in rows
    ]
