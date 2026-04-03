from __future__ import annotations

from typing import Any

from rag_timescale.chunking.base import BaseChunker, Chunk, ChunkResult


class SemanticChunker(BaseChunker):
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> ChunkResult:
        sentences = self._split_sentences(text)
        chunks: list[Chunk] = []
        current_sentences: list[str] = []

        for sentence in sentences:
            current_sentences.append(sentence)
            current_text = " ".join(current_sentences)

            if len(current_text.split()) >= self.chunk_size:
                if len(current_text.split()) > self.max_chunk_size:
                    boundary = self._find_semantic_boundary(current_sentences)
                    if boundary:
                        chunk_text = " ".join(current_sentences[:boundary])
                        remaining = current_sentences[boundary:]
                        if len(chunk_text.split()) >= self.min_chunk_size:
                            chunks.append(
                                Chunk(
                                    content=chunk_text.strip(),
                                    chunk_level=0,
                                    metadata=metadata or {},
                                )
                            )
                        current_sentences = remaining
                    else:
                        chunks.append(
                            Chunk(
                                content=current_text.strip(),
                                chunk_level=0,
                                metadata=metadata or {},
                            )
                        )
                        current_sentences = []
                else:
                    chunks.append(
                        Chunk(
                            content=current_text.strip(),
                            chunk_level=0,
                            metadata=metadata or {},
                        )
                    )
                    current_sentences = []

        if current_sentences:
            remaining = " ".join(current_sentences)
            if len(remaining.split()) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        content=remaining.strip(),
                        chunk_level=0,
                        metadata=metadata or {},
                    )
                )

        return ChunkResult(chunks=chunks)

    def _split_sentences(self, text: str) -> list[str]:
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _find_semantic_boundary(self, sentences: list[str]) -> int | None:
        target = len(sentences) // 2
        for offset in range(min(5, target), -5, -1):
            idx = target + offset
            if 0 < idx < len(sentences):
                sentence = sentences[idx]
                if any(
                    sentence.startswith(w)
                    for w in [
                        "However",
                        "Therefore",
                        "Moreover",
                        "Furthermore",
                        "In addition",
                        "On the other hand",
                        "Consequently",
                        "As a result",
                        "Meanwhile",
                        "Nevertheless",
                    ]
                ):
                    return idx
        return None
