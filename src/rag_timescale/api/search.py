from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Optional

from fastapi import APIRouter
from structlog import get_logger

from rag_timescale.api.deps import RequireKey
from rag_timescale.db.connection import get_pool
from rag_timescale.models import SearchRequest, SearchResponse
from rag_timescale.retrieval.hybrid import (
    get_parent_context,
    hybrid_search,
    hybrid_search_with_rerank,
)

log = get_logger()
router = APIRouter()


@router.post("/", response_model=SearchResponse)
async def search(
    collection_id: uuid.UUID, 
    body: SearchRequest, 
    key_info: RequireKey
) -> SearchResponse:
    """
    Endpoint de recherche principal.
    
    Supporte:
    - Recherche hybride BM25 + vectorielle
    - Reranking optionnel via cross-encoder
    - Contexte parent optionnel
    - Configuration personnalisable par collection
    """
    # Récupération de la configuration de la collection
    config = await _get_config(collection_id)
    
    # Priorité: paramètres de la requête > configuration collection > valeurs par défaut
    bm25_weight = body.bm25_weight if body.bm25_weight is not None else config.get("bm25_weight", 0.5)
    vector_weight = body.vector_weight if body.vector_weight is not None else config.get("vector_weight", 0.5)
    rrf_k = config.get("rrf_k", 60)
    
    # Exécution de la recherche
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
    )
    
    log.info(
        "search_completed",
        collection_id=str(collection_id),
        query=body.query[:50],
        top_k=body.top_k,
        rerank=body.rerank,
        results_count=len(results),
        total_searched=total_searched,
        elapsed_ms=round(elapsed_ms, 2),
    )
    
    return SearchResponse(
        query=body.query,
        results=results,
        total_chunks_searched=total_searched,
        elapsed_ms=round(elapsed_ms, 2),
    )


@router.post("/raw", response_model=SearchResponse)
async def search_raw(
    collection_id: uuid.UUID, 
    body: SearchRequest, 
    key_info: RequireKey
) -> SearchResponse:
    """
    Endpoint de recherche brute sans reranking ni contexte parent.
    
    Plus rapide que l'endpoint principal, idéal pour:
    - Tests rapides
    - Récupération de résultats bruts
    - Scénarios où la latence est critique
    """
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
    )
    
    log.info(
        "search_raw_completed",
        collection_id=str(collection_id),
        query=body.query[:50],
        results_count=len(results),
        elapsed_ms=round(elapsed_ms, 2),
    )
    
    return SearchResponse(
        query=body.query,
        results=results,
        total_chunks_searched=total_searched,
        elapsed_ms=round(elapsed_ms, 2),
    )


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
) -> tuple[list, int, float]:
    """
    Exécute une recherche hybride avec les options spécifiées.
    
    Args:
        collection_id: ID de la collection
        query: Texte de la requête
        top_k: Nombre de résultats à retourner
        rerank: Activer le reranking cross-encoder
        rerank_top_k: Nombre de résultats avant reranking
        include_parent_context: Inclure le contexte parent
        parent_context_levels: Niveaux de parent à remonter
        bm25_weight: Poids BM25 (0-1)
        vector_weight: Poids vectoriel (0-1)
        rrf_k: Paramètre RRF
        filters: Filtres JSONB
        
    Returns:
        Tuple (résultats, total_chunks_recherchés, temps_ms)
    """
    filters = filters or {}
    t_start = time.perf_counter()
    
    # Étape 1: Recherche hybride (avec ou sans rerank)
    if rerank:
        # Utiliser rerank_top_k ou top_k * 2 par défaut
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
    
    # Étape 2: Enrichissement avec contexte parent (optionnel)
    if include_parent_context and results:
        parent_contexts = await _fetch_parent_contexts(
            results, parent_context_levels
        )
        _attach_parent_contexts(results, parent_contexts)
    
    elapsed_ms = (time.perf_counter() - t_start) * 1000
    
    log.debug(
        "execute_search_details",
        rerank=rerank,
        include_parent_context=include_parent_context,
        results_count=len(results),
        total_searched=total_searched,
        elapsed_ms=round(elapsed_ms, 2),
    )
    
    return results, total_searched, elapsed_ms


async def _fetch_parent_contexts(
    results: list, 
    levels: int
) -> list[list[dict[str, Any]]]:
    """
    Récupère les contextes parents pour une liste de résultats.
    
    Args:
        results: Liste des résultats de recherche
        levels: Nombre de niveaux à remonter
        
    Returns:
        Liste des contextes parents pour chaque résultat
    """
    tasks = [
        get_parent_context(chunk_id=r.chunk_id, levels=levels)
        for r in results
    ]
    return await asyncio.gather(*tasks)


def _attach_parent_contexts(
    results: list, 
    parent_contexts: list[list[dict[str, Any]]]
) -> None:
    """
    Attache les contextes parents aux résultats.
    
    Args:
        results: Liste des résultats de recherche
        parent_contexts: Liste des contextes parents correspondants
    """
    for result, parents in zip(results, parent_contexts):
        if parents:
            if not hasattr(result, 'metadata') or result.metadata is None:
                result.metadata = {}
            result.metadata["parent_context"] = parents


async def _get_config(collection_id: uuid.UUID) -> dict[str, Any]:
    """
    Récupère la configuration d'une collection.
    
    Args:
        collection_id: ID de la collection
        
    Returns:
        Dictionnaire de configuration ou {} si non trouvée
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT config FROM collections WHERE id = $1", 
            collection_id
        )
    
    if row and row["config"]:
        return dict(row["config"])
    
    return {}
