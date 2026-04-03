from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    content: str
    chunk_level: int = 0
    section_path: list[str] = field(default_factory=list)
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0

    def __post_init__(self) -> None:
        if self.token_count == 0:
            self.token_count = len(self.content.split())


@dataclass
class ChunkResult:
    chunks: list[Chunk]
    total_tokens: int = 0

    def __post_init__(self) -> None:
        if self.total_tokens == 0:
            self.total_tokens = sum(c.token_count for c in self.chunks)


class BaseChunker(ABC):
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        min_chunk_size: int = 128,
        max_chunk_size: int = 1024,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> ChunkResult:
        pass
