from __future__ import annotations

import asyncio
import numpy as np
from functools import lru_cache
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from structlog import get_logger

from rag_timescale.config import settings

log = get_logger()

_model: SentenceTransformer | None = None
_model_lock = asyncio.Lock()


async def get_model() -> SentenceTransformer:
    """Charge le modèle de manière asynchrone et thread-safe."""
    global _model
    if _model is not None:
        return _model
    async with _model_lock:
        if _model is None:
            # Le chargement du modèle est synchrone, on le fait dans un thread
            loop = asyncio.get_running_loop()
            _model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(
                    settings.embedding.model_name,
                    device=settings.embedding.device,
                )
            )
            log.info("embedding_model_loaded", model=settings.embedding.model_name, device=settings.embedding.device)
    return _model


async def generate_embeddings(
    texts: List[str],
    normalize: Optional[bool] = None,
    batch_size: Optional[int] = None,
) -> List[List[float]]:
    """
    Génère des embeddings pour une liste de textes de manière asynchrone.
    Utilise un thread pool pour ne pas bloquer l'event loop.
    """
    if not texts:
        return []

    model = await get_model()
    if normalize is None:
        normalize = settings.embedding.normalize
    if batch_size is None:
        batch_size = settings.embedding.batch_size

    loop = asyncio.get_running_loop()
    embeddings = await loop.run_in_executor(
        None,
        lambda: model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
    )

    if isinstance(embeddings, np.ndarray):
        return embeddings.tolist()
    return embeddings


async def generate_embedding(text: str) -> List[float]:
    """Génère un embedding pour un texte unique."""
    result = await generate_embeddings([text])
    return result[0] if result else []


@lru_cache(maxsize=10000)
def _cached_embedding_sync(text: str) -> bytes:
    """
    Cache synchrone pour les embeddings fréquents (ex: phrases répétées).
    Retourne un bytes pour être hashable.
    """
    # Utilisation synchrone directe (appelé uniquement dans un thread)
    model = SentenceTransformer(
        settings.embedding.model_name,
        device=settings.embedding.device,
    )
    emb = model.encode([text], normalize_embeddings=settings.embedding.normalize)[0]
    return emb.tobytes()


async def get_cached_embedding(text: str) -> List[float]:
    """
    Version asynchrone avec cache LRU.
    Utile pour les textes répétés (ex: titres de sections identiques).
    """
    loop = asyncio.get_running_loop()
    emb_bytes = await loop.run_in_executor(None, _cached_embedding_sync, text)
    return np.frombuffer(emb_bytes, dtype=np.float32).tolist()


def get_embedding_dimensions() -> int:
    """Retourne la dimension des embeddings (synchrone, appel unique)."""
    model = SentenceTransformer(
        settings.embedding.model_name,
        device=settings.embedding.device,
    )
    return model.get_sentence_embedding_dimension()