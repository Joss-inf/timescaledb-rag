from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from structlog import get_logger

from rag_timescale.config import settings

log = get_logger()
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(
            settings.embedding.model_name,
            device=settings.embedding.device,
        )
        log.info("embedding_model_loaded", model=settings.embedding.model_name, device=settings.embedding.device)
    return _model


def generate_embeddings(texts: list[str], normalize: bool | None = None) -> list[list[float]]:
    if not texts:
        return []

    model = get_model()
    if normalize is None:
        normalize = settings.embedding.normalize

    embeddings = model.encode(
        texts,
        batch_size=settings.embedding.batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )

    if isinstance(embeddings, np.ndarray):
        return embeddings.tolist()
    return embeddings


def generate_embedding(text: str) -> list[float]:
    result = generate_embeddings([text])
    return result[0] if result else []


def get_embedding_dimensions() -> int:
    model = get_model()
    return model.get_sentence_embedding_dimension()
