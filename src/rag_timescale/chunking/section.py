from __future__ import annotations

import asyncio
import re
from typing import Any

from rag_timescale.chunking.base import BaseChunker, Chunk, ChunkResult

_SECTION_RE = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)


class SectionChunker(BaseChunker):
    async def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> ChunkResult:
        """Version asynchrone : exécute le découpage dans un thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._chunk_sync, text, metadata)

    def _chunk_sync(self, text: str, metadata: dict[str, Any] | None) -> ChunkResult:
        """Logique synchrone originale."""
        sections = self._split_by_sections(text)
        chunks: list[Chunk] = []
        base_meta = metadata or {}

        for title, content in sections:
            word_count = len(content.split())
            if word_count > self.max_chunk_size:
                chunks.extend(self._split_content(content, title, base_meta))
            elif word_count >= self.min_chunk_size:
                chunks.append(self._make_chunk(content.strip(), title, base_meta))

        return ChunkResult(chunks=chunks)

    def _split_by_sections(self, text: str) -> list[tuple[str, str]]:
        matches = list(_SECTION_RE.finditer(text))
        if not matches:
            return [("Document", text)]

        sections: list[tuple[str, str]] = []
        for i, match in enumerate(matches):
            title = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            sections.append((title, content))

        return sections

    def _split_content(self, content: str, title: str, metadata: dict[str, Any]) -> list[Chunk]:
        words = content.split()
        step = self.chunk_size - self.overlap
        chunks = []

        for i in range(0, len(words), step):
            segment_words = words[i:i + self.chunk_size]
            segment = " ".join(segment_words)
            if len(segment_words) >= self.min_chunk_size:
                chunks.append(self._make_chunk(segment, title, metadata))

        return chunks

    def _make_chunk(self, content: str, title: str, metadata: dict[str, Any]) -> Chunk:
        return Chunk(
            content=content,
            chunk_level=1,
            section_path=[title],
            metadata=metadata,
        )