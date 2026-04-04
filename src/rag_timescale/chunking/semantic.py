from __future__ import annotations

import numpy as np
from typing import Any

from rag_timescale.chunking.base import BaseChunker, Chunk, ChunkResult
from rag_timescale.embeddings.provider import generate_embedding

class SemanticChunker(BaseChunker):
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> ChunkResult:
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return ChunkResult(chunks=[self._create_chunk(text, metadata)])

        embeddings = [generate_embedding(s) for s in sentences]
        
        distances = []
        for i in range(len(embeddings) - 1):
            similarity = self._cosine_similarity(embeddings[i], embeddings[i+1])
            distances.append(similarity)

        chunks: list[Chunk] = []
        current_sentences: list[str] = [sentences[0]]
        threshold = 0.7 

        for i, similarity in enumerate(distances):
            current_text = " ".join(current_sentences)
            word_count = len(current_text.split())

            if similarity < threshold and word_count >= self.min_chunk_size:
                chunks.append(self._create_chunk(current_text, metadata))
                current_sentences = [sentences[i+1]]
            elif word_count > self.max_chunk_size:
                chunks.append(self._create_chunk(current_text, metadata))
                current_sentences = [sentences[i+1]]
            else:
                current_sentences.append(sentences[i+1])

        if current_sentences:
            chunks.append(self._create_chunk(" ".join(current_sentences), metadata))

        return ChunkResult(chunks=chunks)

    def _split_sentences(self, text: str) -> list[str]:
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return float(dot_product / (norm_v1 * norm_v2))

    def _create_chunk(self, content: str, metadata: dict[str, Any] | None) -> Chunk:
        return Chunk(
            content=content.strip(),
            chunk_level=0,
            metadata=metadata or {},
        )