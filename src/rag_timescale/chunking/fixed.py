from __future__ import annotations

from typing import Any

from rag_timescale.chunking.base import BaseChunker, Chunk, ChunkResult


class FixedChunker(BaseChunker):
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> ChunkResult:
        words = text.split()
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap

        for i in range(0, len(words), step):
            segment = " ".join(words[i : i + self.chunk_size])
            if len(segment.split()) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        content=segment,
                        chunk_level=0,
                        metadata=metadata or {},
                    )
                )

        return ChunkResult(chunks=chunks)
