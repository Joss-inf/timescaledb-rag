from __future__ import annotations

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
) -> BaseChunker:
    cls = _REGISTRY.get(strategy)
    if cls is None:
        raise ValueError(f"Unknown chunking strategy: {strategy}. Available: {list(_REGISTRY.keys())}")
    return cls(
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
    )


def register_chunker(name: str, chunker_cls: type[BaseChunker]) -> None:
    _REGISTRY[name] = chunker_cls


def available_strategies() -> list[str]:
    return list(_REGISTRY.keys())
