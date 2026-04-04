from __future__ import annotations
import json
import uuid
import os
import tempfile
from pathlib import Path
from typing import List

from structlog import get_logger
from fastapi import UploadFile, HTTPException

from rag_timescale.db.connection import get_pool
from rag_timescale.parsers.registry import parse_document
from rag_timescale.chunking.registry import get_chunker
from rag_timescale.embeddings.provider import generate_embeddings
from rag_timescale.models import IngestResponse

log = get_logger()


# ---------------------------
# Batching util
# ---------------------------
def _batched(iterable, size=32):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


# ---------------------------
# MAIN ULTRA INGEST
# ---------------------------
async def process_and_store_optimized(
    collection_id: uuid.UUID,
    file: UploadFile,
    title: str | None,
    external_id: str | None,
    metadata: str | None,
) -> IngestResponse:

    request_id = str(uuid.uuid4())
    doc_metadata = json.loads(metadata) if metadata else {}

    # ---------------------------
    # 1️⃣ Stream vers disque (1 seule fois)
    # ---------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            tmp.write(chunk)
        tmp_path = Path(tmp.name)

    try:
        with open(tmp_path, "rb") as f:
            content = f.read()

        # ---------------------------
        # 2️⃣ Parsing
        # ---------------------------
        parsed = parse_document(content, filename=file.filename, metadata=doc_metadata)

        # ---------------------------
        # 3️⃣ Chunking
        # ---------------------------
        config = await _get_config(collection_id)
        chunker = get_chunker(
            strategy=config.get("chunk_strategy", "hierarchical"),
            chunk_size=config.get("chunk_size", 512),
            overlap=config.get("chunk_overlap", 64),
        )

        chunk_result = chunker.chunk(parsed.text, metadata=parsed.metadata)

        if not chunk_result.chunks:
            raise HTTPException(status_code=400, detail="No chunks generated")

        # ---------------------------
        # 4️⃣ Embeddings batch
        # ---------------------------
        texts = [c.content for c in chunk_result.chunks]
        embeddings: List[list[float]] = []

        for batch in _batched(texts, size=32):
            embeddings.extend(generate_embeddings(batch))

        # ---------------------------
        # 5️⃣ DB transaction
        # ---------------------------
        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():

                # INSERT document
                row = await conn.fetchrow(
                    """INSERT INTO documents 
                    (collection_id, title, external_id, filename, mime_type, size_bytes, metadata) 
                    VALUES ($1,$2,$3,$4,$5,$6,$7)
                    RETURNING id, created_at""",
                    collection_id,
                    title,
                    external_id,
                    file.filename,
                    parsed.mime_type,
                    tmp_path.stat().st_size,
                    parsed.metadata
                )

                doc_id = row["id"]
                created_at = row["created_at"]

                # ---------------------------
                # 6️⃣ COPY chunks (ULTRA FAST)
                # ---------------------------
                records = []

                for chunk, emb in zip(chunk_result.chunks, embeddings, strict=True):

                    parent_uuid = None
                    if chunk.parent_id:
                        try:
                            parent_uuid = uuid.UUID(chunk.parent_id)
                        except Exception:
                            pass

                    records.append((
                        collection_id,
                        doc_id,
                        parent_uuid,
                        chunk.content,
                        emb,
                        chunk.section_path,
                        chunk.chunk_level,
                        chunk.token_count,
                        chunk.metadata
                    ))

                await conn.copy_records_to_table(
                    table_name="chunks",
                    records=records,
                    columns=[
                        "collection_id",
                        "document_id",
                        "parent_id",
                        "content",
                        "embedding",
                        "section_path",
                        "chunk_level",
                        "token_count",
                        "metadata"
                    ]
                )

                # ---------------------------
                # 7️⃣ Update stats
                # ---------------------------
                await conn.execute(
                    """UPDATE documents 
                       SET chunk_count=$1, total_tokens=$2, updated_at=NOW() 
                       WHERE id=$3""",
                    len(chunk_result.chunks),
                    chunk_result.total_tokens,
                    doc_id
                )

        log.info(
            "ingest_completed",
            request_id=request_id,
            doc_id=str(doc_id),
            chunks=len(chunk_result.chunks)
        )

        return IngestResponse(
            document_id=doc_id,
            collection_id=collection_id,
            filename=file.filename,
            chunk_count=len(chunk_result.chunks),
            created_at=created_at
        )

    finally:
        if tmp_path.exists():
            os.unlink(tmp_path)


# ---------------------------
# CONFIG
# ---------------------------
async def _get_config(collection_id: uuid.UUID) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT config FROM collections WHERE id = $1",
            collection_id
        )
    return row["config"] if row and row["config"] else {}