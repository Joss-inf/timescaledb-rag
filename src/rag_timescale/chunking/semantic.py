from __future__ import annotations

import re
import numpy as np
from typing import Any, Optional

from rag_timescale.chunking.base import BaseChunker, Chunk, ChunkResult
from rag_timescale.embeddings.provider import generate_embeddings


class SemanticChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1024,
        min_similarity: float = 0.7,
        batch_size: Optional[int] = None,
    ):
        super().__init__(chunk_size, overlap, min_chunk_size, max_chunk_size)
        self.min_similarity = min_similarity
        self.batch_size = batch_size  # Optionnel, sinon utilise celui du provider

    async def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> ChunkResult:
        sentences = self._split_sentences(text)
        if not sentences:
            return ChunkResult(chunks=[])

        if len(sentences) <= 1:
            return ChunkResult(chunks=[self._create_chunk(text, metadata)])

        # Génération batch de tous les embeddings (un seul appel asynchrone)
        embeddings = await generate_embeddings(sentences, batch_size=self.batch_size)
        # embeddings est une liste de list[float]

        # Calcul des similarités cosinus entre phrases consécutives
        distances = [
            self._cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        chunks: list[Chunk] = []
        current_sentences: list[str] = [sentences[0]]
        current_word_count = self._word_count(sentences[0])

        for i, sim in enumerate(distances):
            next_sentence = sentences[i + 1]
            next_word_count = self._word_count(next_sentence)

            would_exceed = (current_word_count + next_word_count) > self.max_chunk_size

            if (sim < self.min_similarity and current_word_count >= self.min_chunk_size) or would_exceed:
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    if self._word_count(chunk_text) > self.max_chunk_size:
                        chunks.extend(self._split_large_chunk(chunk_text, metadata))
                    else:
                        chunks.append(self._create_chunk(chunk_text, metadata))

                current_sentences = [next_sentence]
                current_word_count = next_word_count
            else:
                current_sentences.append(next_sentence)
                current_word_count += next_word_count

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            if self._word_count(chunk_text) > self.max_chunk_size:
                chunks.extend(self._split_large_chunk(chunk_text, metadata))
            else:
                chunks.append(self._create_chunk(chunk_text, metadata))

        return ChunkResult(chunks=chunks)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
        """Calcule la similarité cosinus entre deux vecteurs (listes Python)."""
        v1_np = np.array(v1)
        v2_np = np.array(v2)
        dot = np.dot(v1_np, v2_np)
        norm1 = np.linalg.norm(v1_np)
        norm2 = np.linalg.norm(v2_np)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))

    def _word_count(self, text: str) -> int:
        return len(text.split())

    def _create_chunk(self, content: str, metadata: dict[str, Any] | None) -> Chunk:
        return Chunk(
            content=content.strip(),
            chunk_level=0,
            metadata=metadata or {},
        )

    def _split_large_chunk(self, content: str, metadata: dict[str, Any] | None) -> list[Chunk]:
        words = content.split()
        step = self.chunk_size - self.overlap
        chunks = []
        for i in range(0, len(words), step):
            segment = " ".join(words[i:i + self.chunk_size])
            if self._word_count(segment) >= self.min_chunk_size:
                chunks.append(self._create_chunk(segment, metadata))
        return chunks