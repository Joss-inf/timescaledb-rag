from __future__ import annotations

import time
import uuid
from fastapi import APIRouter, HTTPException, Query, status

from rag_timescale.api.deps import RequireAdmin
from rag_timescale.auth.permissions import invalidate_collection_cache
from rag_timescale.db.connection import get_pool
from rag_timescale.models import CollectionAccessGrant, PaginatedResponse, decode_cursor, encode_cursor
from structlog import get_logger

log = get_logger()
router = APIRouter()

# Caches simples (TTL 5 secondes)
_collection_exists_cache: dict[str, tuple[float, bool]] = {}
_keys_list_cache: dict[str, tuple[float, tuple[list, str | None]]] = {}
CACHE_TTL = 5.0


async def _collection_exists(collection_id: uuid.UUID) -> bool:
    """Vérifie l'existence d'une collection avec cache court."""
    key = str(collection_id)
    now = time.time()
    if key in _collection_exists_cache:
        ts, exists = _collection_exists_cache[key]
        if now - ts < CACHE_TTL:
            return exists

    pool = await get_pool()
    async with pool.acquire() as conn:
        exists = await conn.fetchval("SELECT 1 FROM collections WHERE id = $1", collection_id)
    exists = bool(exists)
    _collection_exists_cache[key] = (now, exists)
    return exists


def _invalidate_collection_cache(collection_id: str) -> None:
    """Invalide les caches liés à une collection."""
    _collection_exists_cache.pop(collection_id, None)
    _keys_list_cache.pop(collection_id, None)
    invalidate_collection_cache(collection_id)


@router.post("/keys", status_code=status.HTTP_201_CREATED)
async def add_key(
    collection_id: uuid.UUID,
    body: CollectionAccessGrant,
    key_info: RequireAdmin,
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Vérification combinée (clé active + collection)
        check = await conn.fetchrow(
            """
            SELECT 
                (SELECT 1 FROM api_keys WHERE id = $1 AND is_active = TRUE) as key_ok,
                (SELECT 1 FROM collections WHERE id = $2) as coll_ok
            """,
            body.api_key_id,
            collection_id,
        )

        if not check or not check["key_ok"]:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "API key not found or inactive")
        if not check["coll_ok"]:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Collection not found")

        await conn.execute(
            """
            INSERT INTO collection_access (collection_id, api_key_id, access_level)
            VALUES ($1, $2, $3)
            ON CONFLICT (collection_id, api_key_id) DO UPDATE SET access_level = $3
            """,
            collection_id,
            body.api_key_id,
            body.access_level.value,
        )

    _invalidate_collection_cache(str(collection_id))
    log.info("collection_access_granted", collection_id=str(collection_id), api_key_id=str(body.api_key_id))
    return {"message": "API key access granted"}


@router.delete("/keys/{api_key_id}")
async def remove_key(
    collection_id: uuid.UUID,
    api_key_id: uuid.UUID,
    key_info: RequireAdmin,
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM collection_access WHERE collection_id = $1 AND api_key_id = $2",
            collection_id,
            api_key_id,
        )

    if result == "DELETE 0":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Access grant not found")

    _invalidate_collection_cache(str(collection_id))
    log.info("collection_access_revoked", collection_id=str(collection_id), api_key_id=str(api_key_id))
    return {"message": "Access revoked"}


@router.get("/keys", response_model=PaginatedResponse)
async def list_keys(
    collection_id: uuid.UUID,
    key_info: RequireAdmin,
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = None,
):
    # Vérification d'existence avec cache
    if not await _collection_exists(collection_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Collection not found")

    cache_key = f"{collection_id}:{limit}:{cursor}"
    now = time.time()
    if cache_key in _keys_list_cache:
        ts, (items, next_cursor) = _keys_list_cache[cache_key]
        if now - ts < CACHE_TTL:
            return PaginatedResponse(data=items, next_cursor=next_cursor)

    pool = await get_pool()
    async with pool.acquire() as conn:
        if cursor:
            try:
                cursor_granted_at, cursor_key_id = decode_cursor(cursor)
            except ValueError:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid cursor")

            rows = await conn.fetch(
                """
                SELECT ak.id, ak.name, ca.access_level, ca.granted_at
                FROM collection_access ca
                JOIN api_keys ak ON ak.id = ca.api_key_id
                WHERE ca.collection_id = $1
                  AND (ca.granted_at, ak.id) < ($2, $3)
                ORDER BY ca.granted_at DESC, ak.id DESC
                LIMIT $4
                """,
                collection_id,
                cursor_granted_at,
                cursor_key_id,
                limit + 1,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT ak.id, ak.name, ca.access_level, ca.granted_at
                FROM collection_access ca
                JOIN api_keys ak ON ak.id = ca.api_key_id
                WHERE ca.collection_id = $1
                ORDER BY ca.granted_at DESC, ak.id DESC
                LIMIT $2
                """,
                collection_id,
                limit + 1,
            )

    has_more = len(rows) > limit
    items = [
        {
            "key_id": r["id"],
            "name": r["name"],
            "access_level": r["access_level"],
            "granted_at": r["granted_at"],
        }
        for r in (rows[:limit] if has_more else rows)
    ]

    next_cursor = None
    if has_more and items:
        last = items[-1]
        next_cursor = encode_cursor(last["granted_at"], last["key_id"])

    # Mise en cache
    _keys_list_cache[cache_key] = (now, (items, next_cursor))
    return PaginatedResponse(data=items, next_cursor=next_cursor)