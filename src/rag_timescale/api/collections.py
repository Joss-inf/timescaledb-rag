from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query, status

from rag_timescale.api.deps import RequireAdmin, RequireKey, RequireRead
from rag_timescale.auth.permissions import invalidate_collection_cache
from rag_timescale.db.connection import get_pool
from rag_timescale.models import (
    CollectionConfig,
    CollectionCreate,
    CollectionResponse,
    CollectionStats,
    PaginatedResponse,
    decode_cursor,
    encode_cursor,
)

router = APIRouter(prefix="/api/v1/collections", tags=["collections"])


@router.post("", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
async def create_collection(body: CollectionCreate, key_info: RequireKey):
    pool = await get_pool()
    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT 1 FROM collections WHERE name = $1 AND owner_key_id = $2", body.name, key_info["id"]
        )
        if exists:
            raise HTTPException(status_code=409, detail=f"Collection '{body.name}' already exists")
        row = await conn.fetchrow(
            """INSERT INTO collections
            (name, description, config, access_level, owner_key_id)
            VALUES ($1, $2, $3, $4, $5) RETURNING *""",
            body.name,
            body.description,
            body.config.model_dump(),
            body.access_level,
            key_info["id"],
        )
    return row


@router.get("")
async def list_collections(
    key_info: RequireKey,
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
                """SELECT c.* FROM collections c
                WHERE (c.owner_key_id = $1 OR c.access_level = 'public'
                OR EXISTS (SELECT 1 FROM collection_access ca WHERE ca.collection_id = c.id AND ca.api_key_id = $1))
                AND (c.created_at, c.id) < ($2, $3)
                ORDER BY c.created_at DESC, c.id DESC
                LIMIT $4""",
                key_info["id"],
                cursor_created_at,
                cursor_id,
                limit + 1,
            )
        else:
            rows = await conn.fetch(
                """SELECT c.* FROM collections c
                WHERE c.owner_key_id = $1 OR c.access_level = 'public'
                OR EXISTS (SELECT 1 FROM collection_access ca WHERE ca.collection_id = c.id AND ca.api_key_id = $1)
                ORDER BY c.created_at DESC, c.id DESC
                LIMIT $2""",
                key_info["id"],
                limit + 1,
            )
        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]
        next_cursor = None
        if has_more and rows:
            last = rows[-1]
            next_cursor = encode_cursor(last["created_at"], last["id"])
        return PaginatedResponse(data=[dict(r) for r in rows], next_cursor=next_cursor)


@router.get("/{collection_id}")
async def get_collection(collection_id: uuid.UUID, key_info: RequireRead):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM collections WHERE id = $1", collection_id)
    if not row:
        raise HTTPException(status_code=404, detail="Collection not found")
    return row


@router.put("/{collection_id}")
async def update_collection(collection_id: uuid.UUID, body: CollectionCreate, key_info: RequireAdmin):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """UPDATE collections
            SET name=$1, description=$2, config=$3,
            access_level=$4, updated_at=NOW()
            WHERE id=$5 RETURNING *""",
            body.name,
            body.description,
            body.config.model_dump(),
            body.access_level,
            collection_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Collection not found")
    invalidate_collection_cache(str(collection_id))
    return row


@router.delete("/{collection_id}")
async def delete_collection(collection_id: uuid.UUID, key_info: RequireAdmin):
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM collections WHERE id = $1", collection_id)
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Collection not found")
    invalidate_collection_cache(str(collection_id))
    return {"message": "Collection deleted"}


@router.get("/{collection_id}/stats")
async def get_stats(collection_id: uuid.UUID, key_info: RequireRead):
    pool = await get_pool()
    async with pool.acquire() as conn:
        stats = await conn.fetchrow(
            """SELECT c.id as collection_id, COUNT(DISTINCT d.id) as document_count,
            COUNT(ch.id) as chunk_count, COALESCE(SUM(ch.token_count), 0) as total_tokens,
            MAX(d.created_at) as last_ingested_at, c.created_at
            FROM collections c LEFT JOIN documents d ON d.collection_id = c.id
            LEFT JOIN chunks ch ON ch.collection_id = c.id WHERE c.id = $1
            GROUP BY c.id, c.created_at""",
            collection_id,
        )
    if not stats or stats["collection_id"] is None:
        raise HTTPException(status_code=404, detail="Collection not found")
    return CollectionStats(**dict(stats))


@router.put("/{collection_id}/config")
async def update_config(collection_id: uuid.UUID, config: CollectionConfig, key_info: RequireAdmin):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "UPDATE collections SET config = $1, updated_at = NOW() WHERE id = $2 RETURNING *",
            config.model_dump(),
            collection_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Collection not found")
    invalidate_collection_cache(str(collection_id))
    return row
