from __future__ import annotations

import pytest

from rag_timescale.chunking.base import Chunk, ChunkResult
from rag_timescale.chunking.fixed import FixedChunker
from rag_timescale.chunking.registry import get_chunker
from rag_timescale.chunking.hierarchical import HierarchicalChunker
from rag_timescale.chunking.section import SectionChunker
from rag_timescale.chunking.semantic import SemanticChunker


class TestFixedChunker:
    @pytest.mark.asyncio
    async def test_chunk_simple_text(self):
        chunker = FixedChunker(chunk_size=10, overlap=2, min_chunk_size=5)
        text = "This is a test text with many words to chunk properly"
        result = await chunker.chunk(text)

        assert len(result.chunks) > 0
        assert result.total_tokens > 0
        for chunk in result.chunks:
            word_count = len(chunk.content.split())
            assert word_count <= 10 + 2  # allow for overlap

    @pytest.mark.asyncio
    async def test_chunk_with_overlap(self):
        chunker = FixedChunker(chunk_size=5, overlap=2, min_chunk_size=3)
        text = "one two three four five six seven eight nine ten eleven twelve"
        result = await chunker.chunk(text)

        assert len(result.chunks) >= 2

    @pytest.mark.asyncio
    async def test_chunk_empty_text(self):
        chunker = FixedChunker()
        result = await chunker.chunk("")

        assert result.chunks == []
        assert result.total_tokens == 0

    @pytest.mark.asyncio
    async def test_chunk_small_text(self):
        chunker = FixedChunker(chunk_size=100, min_chunk_size=1)
        text = "Short text"
        result = await chunker.chunk(text)

        assert len(result.chunks) >= 1

    @pytest.mark.asyncio
    async def test_chunk_with_metadata(self):
        chunker = FixedChunker(chunk_size=5, min_chunk_size=1)
        text = "word1 word2 word3 word4 word5 word6"
        metadata = {"source": "test", "author": "tester"}
        result = await chunker.chunk(text, metadata=metadata)

        for chunk in result.chunks:
            assert chunk.metadata == metadata


class TestChunkerRegistry:
    def test_get_chunker_fixed(self):
        chunker = get_chunker("fixed", chunk_size=100)
        assert isinstance(chunker, FixedChunker)

    def test_get_chunker_hierarchical(self):
        chunker = get_chunker("hierarchical", chunk_size=100)
        assert isinstance(chunker, HierarchicalChunker)

    def test_get_chunker_section(self):
        chunker = get_chunker("section")
        assert isinstance(chunker, SectionChunker)

    def test_get_chunker_semantic(self):
        chunker = get_chunker("semantic", chunk_size=100)
        assert isinstance(chunker, SemanticChunker)

    def test_get_chunker_default(self):
        chunker = get_chunker("unknown_strategy")
        assert isinstance(chunker, FixedChunker)

    def test_get_chunker_with_params(self):
        chunker = get_chunker("fixed", chunk_size=256, overlap=32, min_chunk_size=64)
        assert chunker.chunk_size == 256
        assert chunker.overlap == 32
        assert chunker.min_chunk_size == 64


class TestChunkModel:
    def test_chunk_creation(self):
        chunk = Chunk(content="test content", chunk_level=0)
        assert chunk.content == "test content"
        assert chunk.chunk_level == 0
        assert chunk.token_count > 0

    def test_chunk_with_parent(self):
        chunk = Chunk(content="child", parent_id="parent-123")
        assert chunk.parent_id == "parent-123"

    def test_chunk_with_section_path(self):
        chunk = Chunk(content="test", section_path=["Chapter 1", "Section 1"])
        assert chunk.section_path == ["Chapter 1", "Section 1"]

    def test_chunk_result(self):
        chunks = [
            Chunk(content="first", token_count=5),
            Chunk(content="second", token_count=5),
        ]
        result = ChunkResult(chunks=chunks)
        assert len(result.chunks) == 2
        assert result.total_tokens == 10


class TestHierarchicalChunker:
    @pytest.mark.asyncio
    async def test_chunk_with_headings(self):
        chunker = HierarchicalChunker(chunk_size=100)
        text = """# Title

This is content under title.

## Section 1

Content in section 1.

### Subsection

More content.
"""
        result = await chunker.chunk(text)
        assert len(result.chunks) > 0

    @pytest.mark.asyncio
    async def test_chunk_without_headings(self):
        chunker = HierarchicalChunker(chunk_size=50)
        text = "Just plain text without any headings at all"
        result = await chunker.chunk(text)
        assert len(result.chunks) > 0


class TestSectionChunker:
    @pytest.mark.asyncio
    async def test_chunk_sections(self):
        chunker = SectionChunker(chunk_size=50, min_chunk_size=1)
        text = "# Section 1\n\nContent in section 1.\n\n# Section 2\n\nContent in section 2."
        result = await chunker.chunk(text)
        assert len(result.chunks) >= 1


class TestSemanticChunker:
    @pytest.mark.asyncio
    async def test_chunk_semantic(self):
        chunker = SemanticChunker(chunk_size=50)
        text = "First semantic unit. Second semantic unit. Third semantic unit."
        result = await chunker.chunk(text)
        assert len(result.chunks) > 0
