from __future__ import annotations

import asyncio
import re
import uuid
from typing import Any

from rag_timescale.chunking.base import BaseChunker, Chunk, ChunkResult

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_MARKDOWN_BLOCK_RE = re.compile(r"^(```|~~~)", re.MULTILINE)
_PAGE_MARKER_RE = re.compile(r"\[\[PAGE_(\d+)\]\]")


class HierarchicalChunker(BaseChunker):
    async def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> ChunkResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._chunk_sync, text, metadata)

    def _chunk_sync(self, text: str, metadata: dict[str, Any] | None = None) -> ChunkResult:
        base_meta = (metadata or {}).copy() if metadata else {}
        sections = self._parse_sections(text)

        chunks: list[Chunk] = []
        section_stack: list[tuple[str, str]] = []  # (title, chunk_id)
        current_path: list[str] = []

        for level, title, clean_content, page_number in sections:
            # Mise à jour du chemin hiérarchique
            while len(current_path) > level - 1:
                current_path.pop()
            current_path.append(title)
            section_path = current_path.copy()

            parent_id = section_stack[-1][1] if section_stack else None

            chunk_meta = base_meta.copy() if base_meta else {}
            chunk_meta["page_number"] = page_number

            word_count = self._word_count(clean_content)
            if word_count <= self.max_chunk_size:
                chunk_id = str(uuid.uuid4())
                chunk = self._make_chunk(
                    content=clean_content,
                    level=level,
                    path=section_path,
                    parent_id=parent_id,
                    metadata=chunk_meta,
                )
                chunk.metadata["_chunk_id"] = chunk_id
                chunks.append(chunk)
            else:
                parent_chunk_id = str(uuid.uuid4())
                parent_chunk = self._make_chunk(
                    content=self._summarize_section(title, clean_content, word_count),
                    level=level,
                    path=section_path,
                    parent_id=parent_id,
                    metadata=chunk_meta,
                )
                parent_chunk.metadata["_chunk_id"] = parent_chunk_id
                chunks.append(parent_chunk)

                leaf_chunks = self._split_leaves(
                    content=clean_content,
                    path=section_path,
                    level=level + 1,
                    parent_id=parent_chunk_id,
                    metadata=chunk_meta,
                )
                chunks.extend(leaf_chunks)
                chunk_id = parent_chunk_id

            section_stack.append((title, chunk_id))

        return ChunkResult(chunks=chunks)

    def _parse_sections(self, text: str) -> list[tuple[int, str, str, int]]:
        """Retourne (level, title, clean_content, page_number)."""
        # Détection des blocs de code
        code_blocks = []
        for match in _MARKDOWN_BLOCK_RE.finditer(text):
            start = match.start()
            end = text.find(match.group(1), start + len(match.group(1)))
            if end != -1:
                code_blocks.append((start, end + len(match.group(1))))

        headings = []
        for match in _HEADING_RE.finditer(text):
            if not any(start <= match.start() <= end for start, end in code_blocks):
                level = len(match.group(1))
                title = match.group(2).strip()
                headings.append((level, title, match.end()))

        if not headings:
            clean_text = _PAGE_MARKER_RE.sub("", text).strip()
            page = self._extract_page_number(text)
            return [(1, "Document", clean_text, page)]

        sections = []
        for i, (level, title, start_pos) in enumerate(headings):
            end_pos = headings[i + 1][2] if i + 1 < len(headings) else len(text)
            raw_content = text[start_pos:end_pos].strip()
            clean_content = _PAGE_MARKER_RE.sub("", raw_content).strip()
            page = self._extract_page_number(raw_content)
            sections.append((level, title, clean_content, page))

        return sections

    @staticmethod
    def _extract_page_number(content: str) -> int:
        match = _PAGE_MARKER_RE.search(content)
        return int(match.group(1)) if match else 1

    @staticmethod
    def _make_chunk(content: str, level: int, path: list[str],
                    parent_id: str | None, metadata: dict) -> Chunk:
        return Chunk(
            content=content,
            chunk_level=level,
            section_path=path,
            parent_id=parent_id,
            metadata=metadata,
        )

    def _word_count(self, text: str) -> int:
        return len(text.split())

    def _summarize_section(self, title: str, content: str, word_count: int) -> str:
        prefix = f"# {title}\n\n"
        if word_count <= self.chunk_size:
            return prefix + content
        words = content.split()
        return prefix + " ".join(words[:self.chunk_size]) + "..."

    def _split_leaves(self, content: str, path: list[str], level: int,
                      parent_id: str, metadata: dict) -> list[Chunk]:
        paragraphs = re.split(r"\n\n+", content)
        chunks = []
        current_paragraphs: list[str] = []
        current_word_count = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_words = para.split()
            para_len = len(para_words)

            if current_word_count + para_len > self.max_chunk_size:
                if current_paragraphs:
                    chunk_text = "\n\n".join(current_paragraphs)
                    chunks.append(self._make_chunk(
                        content=chunk_text,
                        level=level,
                        path=path,
                        parent_id=parent_id,
                        metadata=metadata,
                    ))
                    current_paragraphs = []
                    current_word_count = 0

                if para_len > self.max_chunk_size:
                    chunks.extend(self._split_fixed(para, path, level, parent_id, metadata))
                else:
                    current_paragraphs = [para]
                    current_word_count = para_len
            else:
                current_paragraphs.append(para)
                current_word_count += para_len

        if current_paragraphs:
            chunk_text = "\n\n".join(current_paragraphs)
            chunks.append(self._make_chunk(
                content=chunk_text,
                level=level,
                path=path,
                parent_id=parent_id,
                metadata=metadata,
            ))

        return chunks

    def _split_fixed(self, text: str, path: list[str], level: int,
                     parent_id: str, metadata: dict) -> list[Chunk]:
        words = text.split()
        step = self.chunk_size - self.overlap
        chunks = []
        for i in range(0, len(words), step):
            segment = " ".join(words[i:i + self.chunk_size])
            if len(segment.split()) >= self.min_chunk_size:
                chunks.append(self._make_chunk(
                    content=segment,
                    level=level,
                    path=path,
                    parent_id=parent_id,
                    metadata=metadata,
                ))
        return chunks