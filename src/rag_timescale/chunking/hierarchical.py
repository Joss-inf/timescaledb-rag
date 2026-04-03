from __future__ import annotations

import re
import uuid
from typing import Any

from rag_timescale.chunking.base import BaseChunker, Chunk, ChunkResult

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_MARKDOWN_BLOCK_RE = re.compile(r"^(```|~~~)", re.MULTILINE)


class HierarchicalChunker(BaseChunker):
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> ChunkResult:
        sections = self._parse_sections(text)
        chunks: list[Chunk] = []
        section_stack: list[tuple[str, Chunk]] = []

        for level, title, content in sections:
            section_path = [s[0] for s in section_stack[:level]] + [title]

            if self._content_fits(content):
                chunk = Chunk(
                    content=content.strip(),
                    chunk_level=level,
                    section_path=section_path,
                    parent_id=section_stack[-1][1].section_path[-1] if section_stack else None,
                    metadata=metadata or {},
                )
                chunks.append(chunk)
            else:
                parent_chunk = Chunk(
                    content=self._summarize_section(title, content),
                    chunk_level=level,
                    section_path=section_path,
                    parent_id=section_stack[-1][1].section_path[-1] if section_stack else None,
                    metadata=metadata or {},
                )
                chunks.append(parent_chunk)
                parent_id = str(uuid.uuid4())
                parent_chunk.metadata["_chunk_id"] = parent_id

                leaf_chunks = self._split_leaves(content, section_path, level + 1, parent_id, metadata)
                chunks.extend(leaf_chunks)

            section_stack = section_stack[:level]
            section_stack.append((title, chunks[-1]))

        return ChunkResult(chunks=chunks)

    def _parse_sections(self, text: str) -> list[tuple[int, str, str]]:
        matches = list(_HEADING_RE.finditer(text))
        if not matches:
            return [(0, "Document", text)]

        sections: list[tuple[int, str, str]] = []
        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            sections.append((level, title, content))

        return sections

    def _content_fits(self, content: str) -> bool:
        return len(content.split()) <= self.max_chunk_size

    def _summarize_section(self, title: str, content: str) -> str:
        words = content.split()
        if len(words) <= self.chunk_size:
            return f"# {title}\n\n{content}"
        return f"# {title}\n\n{' '.join(words[: self.chunk_size])}..."

    def _split_leaves(
        self,
        content: str,
        section_path: list[str],
        level: int,
        parent_id: str,
        metadata: dict[str, Any] | None,
    ) -> list[Chunk]:
        paragraphs = re.split(r"\n\n+", content)
        chunks: list[Chunk] = []
        current_text = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len((current_text + "\n\n" + para).split()) > self.max_chunk_size:
                if current_text:
                    chunk = Chunk(
                        content=current_text.strip(),
                        chunk_level=level,
                        section_path=section_path,
                        parent_id=parent_id,
                        metadata=metadata or {},
                    )
                    chunks.append(chunk)

                if len(para.split()) > self.max_chunk_size:
                    chunks.extend(self._split_paragraph(para, section_path, level, parent_id, metadata))
                    current_text = ""
                else:
                    current_text = para
            else:
                current_text = current_text + "\n\n" + para if current_text else para

        if current_text and len(current_text.split()) >= self.min_chunk_size:
            chunks.append(
                Chunk(
                    content=current_text.strip(),
                    chunk_level=level,
                    section_path=section_path,
                    parent_id=parent_id,
                    metadata=metadata or {},
                )
            )

        return chunks

    def _split_paragraph(
        self,
        text: str,
        section_path: list[str],
        level: int,
        parent_id: str,
        metadata: dict[str, Any] | None,
    ) -> list[Chunk]:
        words = text.split()
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap

        for i in range(0, len(words), step):
            segment = " ".join(words[i : i + self.chunk_size])
            if len(segment.split()) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        content=segment,
                        chunk_level=level,
                        section_path=section_path,
                        parent_id=parent_id,
                        metadata=metadata or {},
                    )
                )

        return chunks
