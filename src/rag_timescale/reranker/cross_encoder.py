from __future__ import annotations

from sentence_transformers import CrossEncoder
from structlog import get_logger

from rag_timescale.config import settings

log = get_logger()
_model: CrossEncoder | None = None


def get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(
            settings.reranker.model_name,
            device=settings.reranker.device,
        )
        log.info("reranker_model_loaded", model=settings.reranker.model_name, device=settings.reranker.device)
    return _model


def rerank(query: str, documents: list[str], top_k: int | None = None) -> list[tuple[int, float]]:
    if not documents:
        return []

    if top_k is None:
        top_k = settings.reranker.top_k

    model = get_model()
    pairs = [[query, doc] for doc in documents]
    scores = model.predict(pairs, show_progress_bar=False)

    scored = [(i, float(score)) for i, score in enumerate(scores)]
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]
