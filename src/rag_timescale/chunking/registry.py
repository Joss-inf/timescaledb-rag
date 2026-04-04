from __future__ import annotations

from typing import Any

from rag_timescale.chunking.base import BaseChunker
from rag_timescale.chunking.fixed import FixedChunker
from rag_timescale.chunking.hierarchical import HierarchicalChunker
from rag_timescale.chunking.section import SectionChunker
from rag_timescale.chunking.semantic import SemanticChunker
from rag_timescale.models import ChunkStrategy

_REGISTRY: dict[str, type[BaseChunker]] = {
    ChunkStrategy.HIERARCHICAL: HierarchicalChunker,
    ChunkStrategy.SECTION: SectionChunker,
    ChunkStrategy.SEMANTIC: SemanticChunker,
    ChunkStrategy.FIXED: FixedChunker,
}


def get_chunker(
    strategy: str = "hierarchical",
    chunk_size: int = 512,
    overlap: int = 64,
    min_chunk_size: int = 128,
    max_chunk_size: int = 1024,
    **kwargs: Any,
) -> BaseChunker:
    """
    Retourne une instance du chunker demandé, configurée avec les paramètres donnés.

    Args:
        strategy: Nom de la stratégie ('hierarchical', 'section', 'semantic', 'fixed')
        chunk_size: Taille de base des chunks (en mots)
        overlap: Chevauchement entre chunks (en mots)
        min_chunk_size: Taille minimale d'un chunk
        max_chunk_size: Taille maximale d'un chunk
        **kwargs: Paramètres supplémentaires spécifiques au chunker (ex: min_similarity pour SemanticChunker)

    Returns:
        Instance de BaseChunker (asynchrone)
    """
    cls = _REGISTRY.get(strategy)
    if cls is None:
        cls = FixedChunker  # Default fallback

    # Fusion des paramètres par défaut avec les kwargs supplémentaires
    params = {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "min_chunk_size": min_chunk_size,
        "max_chunk_size": max_chunk_size,
        **kwargs,
    }
    return cls(**params)


def register_chunker(name: str, chunker_cls: type[BaseChunker]) -> None:
    """Enregistre une nouvelle stratégie de chunking."""
    _REGISTRY[name] = chunker_cls


def available_strategies() -> list[str]:
    """Retourne la liste des stratégies disponibles."""
    return list(_REGISTRY.keys())
