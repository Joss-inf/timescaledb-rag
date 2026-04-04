from __future__ import annotations

import asyncio
from typing import Any

from rag_timescale.chunking.base import BaseChunker, Chunk, ChunkResult


class FixedChunker(BaseChunker):
    async def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> ChunkResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._chunk_sync, text, metadata)

    def _chunk_sync(self, text: str, metadata: dict[str, Any] | None = None) -> ChunkResult:
        words = text.split()
        num_words = len(words)
        chunks: list[Chunk] = []
    
        step = max(1, self.chunk_size - self.overlap)
    
        for i in range(0, num_words, step):
            word_slice = words[i : i + self.chunk_size]
            slice_len = len(word_slice)
            
            if slice_len >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        content=" ".join(word_slice),
                        chunk_level=0,
                        metadata=metadata or {},
                    )
                )

            if i + step >= num_words:
                break

        return ChunkResult(chunks=chunks)