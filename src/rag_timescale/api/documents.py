from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Form, HTTPException, Query, UploadFile, status

from rag_timescale.api.deps import RequireAdmin, RequireRead, RequireWrite
from rag_timescale.chunking.registry import get_chunker
from rag_timescale.db.connection import get_pool
from rag_timescale.embeddings.provider import generate_embeddings
from rag_timescale.models import IngestResponse, PaginatedResponse, decode_cursor, encode_cursor
from rag_timescale.parsers.registry import parse_document

router = APIRouter(prefix="/api/v1/collections", tags=["documents"])

DOC_FIELDS = (
    "id, external_id, title, filename, mime_type, size_bytes, "
    "chunk_count, total_tokens, metadata, created_at, updated_at"
)


@router.post("/{collection_id}/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(
    collection_id: uuid.UUID,
    key_info: RequireWrite,
    file: UploadFile,
    title: str | None = Form(None),
    external_id: str | None = Form(None),
    metadata: str | None = Form(None),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    return await _process_and_store(collection_id, content, file.filename or "unknown", title, external_id, metadata)


@router.post("/{collection_id}/batch/ingest")
async def batch_ingest(
    collection_id: uuid.UUID,
    key_info: RequireWrite,
    files: list[UploadFile],
):
    config = await _get_config(collection_id)
    chunker = _make_chunker(config)
    pool = await get_pool()
    results = []
    for file in files:
        content = await file.read()
        if not content:
            continue
        try:
            doc_id = await _store_document(
                collection_id, content, file.filename or "unknown", None, None, config, chunker, pool
            )
            results.append({"filename": file.filename, "document_id": str(doc_id), "status": "ok"})
        except Exception as e:
            results.append({"filename": file.filename, "status": "error", "detail": str(e)})
    return {"results": results, "total": len(results)}


@router.post("/{collection_id}/documents/{document_id}/update", response_model=IngestResponse)
async def update_document(
    collection_id: uuid.UUID,
    document_id: uuid.UUID,
    key_info: RequireWrite,
    file: UploadFile,
    title: str | None = Form(None),
    external_id: str | None = Form(None),
    metadata: str | None = Form(None),
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT id FROM documents WHERE id = $1 AND collection_id = $2", document_id, collection_id
        )
        if not existing:
            raise HTTPException(status_code=404, detail="Document not found")
        await conn.execute("DELETE FROM chunks WHERE document_id = $1", document_id)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    return await _process_and_store(
        collection_id, content, file.filename or "unknown", title, external_id, metadata, document_id
    )


@router.delete("/{collection_id}/documents/{document_id}")
async def delete_document(collection_id: uuid.UUID, document_id: uuid.UUID, key_info: RequireAdmin):
    pool = await get_pool()
    async with pool.acquire() as conn:
        doc = await conn.fetchrow(
            "SELECT id, title, filename FROM documents WHERE id = $1 AND collection_id = $2", document_id, collection_id
        )
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        await conn.execute("DELETE FROM documents WHERE id = $1", document_id)
    return {"message": "Document deleted", "deleted": {"id": doc["id"], "title": doc["title"] or doc["filename"]}}


@router.delete("/{collection_id}/documents/by-external-id/{external_id}")
async def delete_document_by_external_id(collection_id: uuid.UUID, external_id: str, key_info: RequireAdmin):
    pool = await get_pool()
    async with pool.acquire() as conn:
        doc = await conn.fetchrow(
            "SELECT id, title, filename FROM documents WHERE collection_id = $1 AND external_id = $2",
            collection_id,
            external_id,
        )
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        await conn.execute("DELETE FROM documents WHERE id = $1", doc["id"])
    return {
        "message": "Document deleted",
        "deleted": {"id": doc["id"], "external_id": external_id, "title": doc["title"] or doc["filename"]},
    }


@router.get("/{collection_id}/documents")
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
                raise HTTPException(status_code=400, detail="Invalid cursor") from None
            rows = await conn.fetch(
                f"""SELECT {DOC_FIELDS} FROM documents
                WHERE collection_id = $1 AND (created_at, id) < ($2, $3)
                ORDER BY created_at DESC, id DESC
                LIMIT $4""",
                collection_id,
                cursor_created_at,
                cursor_id,
                limit + 1,
            )
        else:
            rows = await conn.fetch(
                f"SELECT {DOC_FIELDS} FROM documents WHERE collection_id = $1 "
                "ORDER BY created_at DESC, id DESC LIMIT $2",
                collection_id,
                limit + 1,
            )
        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]
        next_cursor = None
        if has_more and rows:
            last = rows[-1]
            next_cursor = encode_cursor(last["created_at"], last["id"])
        return PaginatedResponse(data=[_doc_row_to_dict(r) for r in rows], next_cursor=next_cursor)


@router.get("/{collection_id}/documents/by-external-id/{external_id}")
async def get_document_by_external_id(
    collection_id: uuid.UUID, external_id: str, key_info: RequireRead, include_chunks: bool = False
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        doc = await conn.fetchrow(
            f"SELECT {DOC_FIELDS} FROM documents WHERE collection_id = $1 AND external_id = $2",
            collection_id,
            external_id,
        )
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document with external_id '{external_id}' not found")
        result = _doc_row_to_dict(doc)
        if include_chunks:
            chunks = await conn.fetch(
                """SELECT id, content, section_path, chunk_level, token_count
                FROM chunks WHERE document_id = $1 ORDER BY chunk_level, id""",
                doc["id"],
            )
            result["chunks"] = [dict(c) for c in chunks]
    return result


@router.get("/{collection_id}/documents/{document_id}")
async def get_document(
    collection_id: uuid.UUID, document_id: uuid.UUID, key_info: RequireRead, include_chunks: bool = False
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        doc = await conn.fetchrow(
            f"SELECT {DOC_FIELDS} FROM documents WHERE id = $1 AND collection_id = $2", document_id, collection_id
        )
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        result = _doc_row_to_dict(doc)
        if include_chunks:
            chunks = await conn.fetch(
                """SELECT id, content, section_path, chunk_level, token_count
                FROM chunks WHERE document_id = $1 ORDER BY chunk_level, id""",
                document_id,
            )
            result["chunks"] = [dict(c) for c in chunks]
    return result


@router.get("/{collection_id}/chunks/{chunk_id}")
async def get_chunk(
    collection_id: uuid.UUID,
    chunk_id: uuid.UUID,
    key_info: RequireRead,
    include_parent: bool = False,
    parent_levels: int = 1,
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        chunk = await conn.fetchrow(
            """SELECT id, document_id, parent_id, content, section_path,
            chunk_level, token_count, metadata FROM chunks
            WHERE id = $1 AND collection_id = $2""",
            chunk_id,
            collection_id,
        )
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")
        result = dict(chunk)
        if include_parent and result.get("parent_id"):
            from rag_timescale.retrieval.hybrid import get_parent_context

            result["parent_context"] = await get_parent_context(chunk_id, levels=parent_levels)
    return result


def _doc_row_to_dict(r) -> dict:
    return {
        "id": r["id"],
        "external_id": r["external_id"],
        "title": r["title"] or r["filename"],
        "filename": r["filename"],
        "mime_type": r["mime_type"],
        "size_bytes": r["size_bytes"],
        "chunk_count": r["chunk_count"],
        "total_tokens": r["total_tokens"],
        "metadata": r["metadata"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
    }


async def _process_and_store(
    collection_id: uuid.UUID,
    content: bytes,
    filename: str,
    title: str | None,
    external_id: str | None,
    metadata: str | None,
    doc_id: uuid.UUID | None = None,
) -> IngestResponse:
    doc_metadata = json.loads(metadata) if metadata else {}
    doc_metadata["filename"] = filename
    parsed = parse_document(content, filename=filename, metadata=doc_metadata)
    config = await _get_config(collection_id)
    chunker = _make_chunker(config)
    chunk_result = chunker.chunk(parsed.text, metadata=parsed.metadata)
    if not chunk_result.chunks:
        raise HTTPException(status_code=400, detail="No chunks generated")
    embeddings = generate_embeddings([c.content for c in chunk_result.chunks])
    pool = await get_pool()
    async with pool.acquire() as conn:
        if doc_id is None:
            row = await conn.fetchrow(
                """INSERT INTO documents (collection_id, title, external_id, filename, mime_type, size_bytes, metadata)
                VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING id, created_at""",
                collection_id,
                title,
                external_id,
                filename,
                parsed.mime_type,
                len(content),
                parsed.metadata,
            )
            doc_id, created_at = row["id"], row["created_at"]
        else:
            row = await conn.fetchrow(
                """UPDATE documents SET title=$1, external_id=$2, filename=$3, mime_type=$4, size_bytes=$5,
                metadata=$6, updated_at=NOW() WHERE id=$7 RETURNING created_at""",
                title,
                external_id,
                filename,
                parsed.mime_type,
                len(content),
                parsed.metadata,
                doc_id,
            )
            created_at = row["created_at"]
        for chunk, emb in zip(chunk_result.chunks, embeddings, strict=True):
            parent_uuid = (
                uuid.UUID(chunk.parent_id) if chunk.parent_id and chunk.parent_id != chunk.section_path[-1] else None
            )
            await conn.execute(
                """INSERT INTO chunks (collection_id, document_id, parent_id, content, embedding,
                section_path, chunk_level, token_count, metadata) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)""",
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
        await conn.execute(
            "UPDATE documents SET chunk_count=$1, total_tokens=$2, updated_at=NOW() WHERE id=$3",
            len(chunk_result.chunks),
            chunk_result.total_tokens,
            doc_id,
        )
    return IngestResponse(
        document_id=doc_id,
        collection_id=collection_id,
        filename=filename,
        chunk_count=len(chunk_result.chunks),
        created_at=created_at,
    )


async def _store_document(
    collection_id: uuid.UUID,
    content: bytes,
    filename: str,
    title: str | None,
    external_id: str | None,
    config: dict,
    chunker,
    pool,
) -> uuid.UUID:
    parsed = parse_document(content, filename=filename, metadata={"filename": filename})
    chunk_result = chunker.chunk(parsed.text, metadata=parsed.metadata)
    if not chunk_result.chunks:
        raise ValueError("No chunks generated")
    embeddings = generate_embeddings([c.content for c in chunk_result.chunks])
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO documents (collection_id, title, external_id, filename, mime_type, size_bytes, metadata)
            VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING id""",
            collection_id,
            title,
            external_id,
            filename,
            parsed.mime_type,
            len(content),
            parsed.metadata,
        )
        doc_id = row["id"]
        for chunk, emb in zip(chunk_result.chunks, embeddings, strict=True):
            parent_uuid = (
                uuid.UUID(chunk.parent_id) if chunk.parent_id and chunk.parent_id != chunk.section_path[-1] else None
            )
            await conn.execute(
                """INSERT INTO chunks (collection_id, document_id, parent_id, content, embedding,
                section_path, chunk_level, token_count, metadata) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)""",
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
        await conn.execute(
            "UPDATE documents SET chunk_count=$1, total_tokens=$2, updated_at=NOW() WHERE id=$3",
            len(chunk_result.chunks),
            chunk_result.total_tokens,
            doc_id,
        )
    return doc_id


async def _get_config(collection_id: uuid.UUID) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT config FROM collections WHERE id = $1", collection_id)
    return row["config"] if row and row["config"] else {}


def _make_chunker(config: dict):
    return get_chunker(
        strategy=config.get("chunk_strategy", "hierarchical"),
        chunk_size=config.get("chunk_size", 512),
        overlap=config.get("chunk_overlap", 64),
        min_chunk_size=config.get("min_chunk_size", 128),
        max_chunk_size=config.get("max_chunk_size", 1024),
    )
