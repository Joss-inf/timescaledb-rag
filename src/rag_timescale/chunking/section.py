from __future__ import annotations

import re
from typing import Any

from rag_timescale.chunking.base import BaseChunker, Chunk, ChunkResult

_SECTION_RE = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)


class SectionChunker(BaseChunker):
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> ChunkResult:
        sections = self._split_by_sections(text)
        chunks: list[Chunk] = []

        for title, content in sections:
            if len(content.split()) > self.max_chunk_size:
                leaf_chunks = self._split_content(content, title, metadata)
                chunks.extend(leaf_chunks)
            elif len(content.split()) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        content=content.strip(),
                        chunk_level=1,
                        section_path=[title],
                        metadata=metadata or {},
                    )
                )

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

    def _split_content(self, content: str, title: str, metadata: dict[str, Any] | None) -> list[Chunk]:
        words = content.split()
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap

        for i in range(0, len(words), step):
            segment = " ".join(words[i : i + self.chunk_size])
            if len(segment.split()) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        content=segment,
                        chunk_level=1,
                        section_path=[title],
                        metadata=metadata or {},
                    )
                )

        return chunks
