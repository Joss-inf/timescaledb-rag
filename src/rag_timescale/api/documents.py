from __future__ import annotations
import json
import uuid
import os
import tempfile
from pathlib import Path
from typing import List

from structlog import get_logger
from fastapi import APIRouter, UploadFile, HTTPException, Query, Form, status
from fastapi.responses import StreamingResponse

from rag_timescale.api.deps import RequireRead, RequireWrite, RequireAdmin
from rag_timescale.db.connection import get_pool
from rag_timescale.parsers.registry import parse_document
from rag_timescale.chunking.registry import get_chunker
from rag_timescale.embeddings.provider import generate_embeddings
from rag_timescale.models import (
    IngestResponse,
    DocumentResponse,
    ChunkResponse,
    PaginatedResponse,
    decode_cursor,
    encode_cursor,
)

log = get_logger()
router = APIRouter()


# ---------------------------
# Batching util
# ---------------------------
def _batched(iterable, size=32):
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


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
                    parsed.metadata,
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

                    records.append(
                        (
                            collection_id,
                            doc_id,
                            parent_uuid,
                            chunk.content,
                            emb,
                            chunk.section_path,
                            chunk.chunk_level,
                            chunk.token_count,
                            chunk.metadata,
                        )
                    )

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
                        "metadata",
                    ],
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
                    doc_id,
                )

        log.info("ingest_completed", request_id=request_id, doc_id=str(doc_id), chunks=len(chunk_result.chunks))

        return IngestResponse(
            document_id=doc_id,
            collection_id=collection_id,
            filename=file.filename,
            chunk_count=len(chunk_result.chunks),
            created_at=created_at,
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
        row = await conn.fetchrow("SELECT config FROM collections WHERE id = $1", collection_id)
    return row["config"] if row and row["config"] else {}


# ---------------------------
# ENDPOINTS
# ---------------------------


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(
    collection_id: uuid.UUID,
    file: UploadFile,
    key_info: RequireWrite,
    title: str | None = Form(None),
    external_id: str | None = Form(None),
    metadata: str | None = Form(None),
):
    return await process_and_store_optimized(
        collection_id=collection_id,
        file=file,
        title=title,
        external_id=external_id,
        metadata=metadata,
    )


@router.post("/batch/ingest", response_model=list[IngestResponse], status_code=status.HTTP_201_CREATED)
async def batch_ingest(
    collection_id: uuid.UUID,
    files: List[UploadFile],
    key_info: RequireWrite,
):
    results = []
    for file in files:
        result = await process_and_store_optimized(
            collection_id=collection_id,
            file=file,
            title=file.filename,
            external_id=None,
            metadata=None,
        )
        results.append(result)
    return results


@router.get("/", response_model=PaginatedResponse)
async def list_documents(
    collection_id: uuid.UUID,
    key_info: RequireRead,
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = None,
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        if cursor:
            try:
                cursor_created_at, cursor_id = decode_cursor(cursor)
            except ValueError:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid cursor")

            rows = await conn.fetch(
                """SELECT * FROM documents 
                WHERE collection_id = $1 AND deleted_at IS NULL
                AND (created_at, id) < ($2, $3)
                ORDER BY created_at DESC, id DESC
                LIMIT $4""",
                collection_id,
                cursor_created_at,
                cursor_id,
                limit + 1,
            )
        else:
            rows = await conn.fetch(
                """SELECT * FROM documents 
                WHERE collection_id = $1 AND deleted_at IS NULL
                ORDER BY created_at DESC, id DESC
                LIMIT $2""",
                collection_id,
                limit + 1,
            )

    has_more = len(rows) > limit
    items = [dict(r) for r in (rows[:limit] if has_more else rows)]

    next_cursor = None
    if has_more and items:
        last = items[-1]
        next_cursor = encode_cursor(last["created_at"], last["id"])

    return PaginatedResponse(data=items, next_cursor=next_cursor)


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    collection_id: uuid.UUID,
    document_id: uuid.UUID,
    key_info: RequireRead,
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM documents WHERE id = $1 AND collection_id = $2 AND deleted_at IS NULL",
            document_id,
            collection_id,
        )

    if not row:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")

    return dict(row)


@router.get("/by-external-id/{external_id}", response_model=DocumentResponse)
async def get_document_by_external_id(
    collection_id: uuid.UUID,
    external_id: str,
    key_info: RequireRead,
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM documents WHERE external_id = $1 AND collection_id = $2 AND deleted_at IS NULL",
            external_id,
            collection_id,
        )

    if not row:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")

    return dict(row)


@router.post("/{document_id}/update", response_model=IngestResponse)
async def update_document(
    collection_id: uuid.UUID,
    document_id: uuid.UUID,
    file: UploadFile,
    key_info: RequireWrite,
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT id FROM documents WHERE id = $1 AND collection_id = $2", document_id, collection_id
        )

    if not existing:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")

    await conn.execute("UPDATE documents SET deleted_at = NOW() WHERE id = $1", document_id)

    result = await process_and_store_optimized(
        collection_id=collection_id,
        file=file,
        title=file.filename,
        external_id=None,
        metadata=None,
    )

    return result


@router.delete("/{document_id}")
async def delete_document(
    collection_id: uuid.UUID,
    document_id: uuid.UUID,
    key_info: RequireAdmin,
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE documents SET deleted_at = NOW() WHERE id = $1 AND collection_id = $2", document_id, collection_id
        )

    if result == "UPDATE 0":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")

    await conn.execute("UPDATE chunks SET deleted_at = NOW() WHERE document_id = $1", document_id)

    return {"message": "Document deleted"}


@router.delete("/by-external-id/{external_id}")
async def delete_document_by_external_id(
    collection_id: uuid.UUID,
    external_id: str,
    key_info: RequireAdmin,
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        doc = await conn.fetchrow(
            "SELECT id FROM documents WHERE external_id = $1 AND collection_id = $2", external_id, collection_id
        )

        if not doc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")

        await conn.execute("UPDATE documents SET deleted_at = NOW() WHERE id = $1", doc["id"])
        await conn.execute("UPDATE chunks SET deleted_at = NOW() WHERE document_id = $1", doc["id"])

    return {"message": "Document deleted"}


@router.get("/{document_id}/chunks/{chunk_id}", response_model=ChunkResponse)
async def get_chunk(
    collection_id: uuid.UUID,
    document_id: uuid.UUID,
    chunk_id: uuid.UUID,
    key_info: RequireRead,
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT * FROM chunks 
            WHERE id = $1 AND document_id = $2 AND collection_id = $3 AND deleted_at IS NULL""",
            chunk_id,
            document_id,
            collection_id,
        )

    if not row:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Chunk not found")

    return dict(row)
