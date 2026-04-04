from __future__ import annotations

import asyncio
from functools import lru_cache
from sentence_transformers import CrossEncoder
from structlog import get_logger

from rag_timescale.config import settings

log = get_logger()
_model: CrossEncoder | None = None
_model_lock = asyncio.Lock()


async def get_model() -> CrossEncoder:
    """Charge le modèle CrossEncoder de manière asynchrone et thread-safe."""
    global _model
    if _model is not None:
        return _model
    async with _model_lock:
        if _model is None:
            loop = asyncio.get_running_loop()
            _model = await loop.run_in_executor(
                None,
                lambda: CrossEncoder(
                    settings.reranker.model_name,
                    device=settings.reranker.device,
                )
            )
            log.info("reranker_model_loaded", model=settings.reranker.model_name, device=settings.reranker.device)
    return _model


@lru_cache(maxsize=10000)
def _cached_score_sync(query: str, doc: str) -> float:
    """
    Version synchrone avec cache LRU pour une paire (query, document).
    Utile si les mêmes paires apparaissent souvent.
    """
    model = CrossEncoder(
        settings.reranker.model_name,
        device=settings.reranker.device,
    )
    return float(model.predict([[query, doc]], show_progress_bar=False)[0])


async def rerank(
    query: str,
    documents: list[str],
    top_k: int | None = None,
    use_cache: bool = False,
) -> list[tuple[int, float]]:
    """
    Reranke les documents par rapport à la query en utilisant un CrossEncoder.

    Args:
        query: La requête
        documents: Liste des textes à reranker
        top_k: Nombre de résultats à retourner (défaut: config.reranker.top_k)
        use_cache: Utiliser un cache LRU pour les paires (défaut: False)

    Returns:
        Liste de tuples (index_document, score) triée par score décroissant
    """
    if not documents:
        return []

    if top_k is None:
        top_k = settings.reranker.top_k

    model = await get_model()
    loop = asyncio.get_running_loop()

    if use_cache:
        # Cache par paire (lent à construire pour de gros lots, mais utile pour les répétitions)
        scores = []
        for doc in documents:
            score = await loop.run_in_executor(None, _cached_score_sync, query, doc)
            scores.append(score)
    else:
        # Batch processing classique
        pairs = [[query, doc] for doc in documents]
        scores = await loop.run_in_executor(
            None,
            lambda: model.predict(pairs, show_progress_bar=False)
        )
        scores = [float(s) for s in scores]

    scored = list(enumerate(scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    log.debug("rerank_completed", query=query[:50], n_docs=len(documents), top_k=top_k, use_cache=use_cache)
    return scored[:top_k]


async def rerank_single(query: str, document: str) -> float:
    """Reranke un seul document (utile pour les tests ou calculs isolés)."""
    result = await rerank(query, [document], top_k=1)
    return result[0][1] if result else 0.0