from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query, status

from rag_timescale.api.deps import RequireAdmin
from rag_timescale.auth.permissions import invalidate_collection_cache
from rag_timescale.db.connection import get_pool
from rag_timescale.models import CollectionAccessGrant, PaginatedResponse, decode_cursor, encode_cursor

router = APIRouter(prefix="/api/v1/collections", tags=["permissions"])


@router.post("/{collection_id}/keys", status_code=status.HTTP_201_CREATED)
async def add_key(collection_id: uuid.UUID, body: CollectionAccessGrant, key_info: RequireAdmin):
    pool = await get_pool()
    async with pool.acquire() as conn:
        key_exists = await conn.fetchval("SELECT 1 FROM api_keys WHERE id = $1 AND is_active = TRUE", body.api_key_id)
        if not key_exists:
            raise HTTPException(status_code=404, detail="API key not found or inactive")
        await conn.execute(
            """INSERT INTO collection_access (collection_id, api_key_id, access_level)
            VALUES ($1, $2, $3)
            ON CONFLICT (collection_id, api_key_id) DO UPDATE SET access_level = $3""",
            collection_id,
            body.api_key_id,
            body.access_level.value,
        )
    invalidate_collection_cache(str(collection_id))
    return {"message": "API key added to collection"}


@router.delete("/{collection_id}/keys/{api_key_id}")
async def remove_key(collection_id: uuid.UUID, api_key_id: uuid.UUID, key_info: RequireAdmin):
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM collection_access WHERE collection_id = $1 AND api_key_id = $2",
            collection_id,
            api_key_id,
        )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Key access not found")
    invalidate_collection_cache(str(collection_id))
    return {"message": "API key removed from collection"}


@router.get("/{collection_id}/keys")
async def list_keys(
    collection_id: uuid.UUID,
    key_info: RequireAdmin,
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = None,
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        if cursor:
            try:
                cursor_granted_at, cursor_key_id = decode_cursor(cursor)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid cursor") from None
            rows = await conn.fetch(
                """SELECT ak.id, ak.name, ca.access_level, ca.granted_at
                FROM collection_access ca JOIN api_keys ak ON ak.id = ca.api_key_id
                WHERE ca.collection_id = $1 AND (ca.granted_at, ak.id) < ($2, $3)
                ORDER BY ca.granted_at DESC, ak.id DESC
                LIMIT $4""",
                collection_id,
                cursor_granted_at,
                cursor_key_id,
                limit + 1,
            )
        else:
            rows = await conn.fetch(
                """SELECT ak.id, ak.name, ca.access_level, ca.granted_at
                FROM collection_access ca JOIN api_keys ak ON ak.id = ca.api_key_id
                WHERE ca.collection_id = $1 ORDER BY ca.granted_at DESC, ak.id DESC
                LIMIT $2""",
                collection_id,
                limit + 1,
            )
        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]
        next_cursor = None
        if has_more and rows:
            last = rows[-1]
            next_cursor = encode_cursor(last["granted_at"], last["id"])
        return PaginatedResponse(
            data=[
                {"key_id": r["id"], "name": r["name"], "access_level": r["access_level"], "granted_at": r["granted_at"]}
                for r in rows
            ],
            next_cursor=next_cursor,
        )
