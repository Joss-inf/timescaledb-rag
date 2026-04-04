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
    CollectionUpdate,
    PaginatedResponse,
    decode_cursor,
    encode_cursor,
)

router = APIRouter()

# Cache simple pour get_collection (optionnel)
_collection_cache: dict[str, tuple[float, dict]] = {}
CACHE_TTL = 5  # secondes


async def _get_cached_collection(collection_id: str) -> dict | None:
    import time
    now = time.time()
    if collection_id in _collection_cache:
        ts, data = _collection_cache[collection_id]
        if now - ts < CACHE_TTL:
            return data
    return None


def _set_cached_collection(collection_id: str, data: dict):
    import time
    _collection_cache[collection_id] = (time.time(), data)


# ---------------------------
# CREATE
# ---------------------------
@router.post("/", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
async def create_collection(body: CollectionCreate, key_info: RequireKey):
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            exists = await conn.fetchval(
                "SELECT 1 FROM collections WHERE name = $1 AND owner_key_id = $2",
                body.name, key_info["id"]
            )
            if exists:
                raise HTTPException(status.HTTP_409_CONFLICT, f"Collection '{body.name}' already exists")

            row = await conn.fetchrow(
                """INSERT INTO collections 
                   (name, description, config, access_level, owner_key_id)
                   VALUES ($1, $2, $3, $4, $5)
                   RETURNING *""",
                body.name,
                body.description,
                body.config.model_dump(),
                body.access_level,
                key_info["id"],
            )
    return dict(row)


# ---------------------------
# LIST (OPTIMIZED)
# ---------------------------
@router.get("/", response_model=PaginatedResponse)
async def list_collections(
    key_info: RequireKey,
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = None,
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Utilisation de UNION pour éviter DISTINCT (plus efficace)
        if cursor:
            try:
                cursor_created_at, cursor_id = decode_cursor(cursor)
            except ValueError:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid cursor")

            rows = await conn.fetch(
                """
                (SELECT * FROM collections WHERE owner_key_id = $1)
                UNION
                (SELECT c.* FROM collections c
                 JOIN collection_access ca ON ca.collection_id = c.id
                 WHERE ca.api_key_id = $1)
                UNION
                (SELECT * FROM collections WHERE access_level = 'public')
                ORDER BY created_at DESC, id DESC
                LIMIT $2
                """,
                key_info["id"],
                limit + 1
            )
            # Appliquer le curseur après (plus simple)
            filtered = []
            for r in rows:
                if (r["created_at"], r["id"]) < (cursor_created_at, cursor_id):
                    filtered.append(r)
            rows = filtered[:limit+1]
        else:
            rows = await conn.fetch(
                """
                (SELECT * FROM collections WHERE owner_key_id = $1)
                UNION
                (SELECT c.* FROM collections c
                 JOIN collection_access ca ON ca.collection_id = c.id
                 WHERE ca.api_key_id = $1)
                UNION
                (SELECT * FROM collections WHERE access_level = 'public')
                ORDER BY created_at DESC, id DESC
                LIMIT $2
                """,
                key_info["id"],
                limit + 1
            )

        has_more = len(rows) > limit
        items = [dict(r) for r in rows[:limit]]
        next_cursor = None
        if has_more and items:
            last = items[-1]
            next_cursor = encode_cursor(last["created_at"], last["id"])

        return PaginatedResponse(data=items, next_cursor=next_cursor)


# ---------------------------
# GET (SECURED)
# ---------------------------
@router.get("/{collection_id}")
async def get_collection(collection_id: uuid.UUID, key_info: RequireRead):
    # Cache optionnel
    cached = await _get_cached_collection(str(collection_id))
    if cached:
        return cached

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT c.* FROM collections c
            LEFT JOIN collection_access ca
                ON ca.collection_id = c.id AND ca.api_key_id = $2
            WHERE c.id = $1
              AND (
                  c.owner_key_id = $2
                  OR c.access_level = 'public'
                  OR ca.api_key_id IS NOT NULL
              )
            """,
            collection_id,
            key_info["id"]
        )

    if not row:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Collection not found")

    data = dict(row)
    _set_cached_collection(str(collection_id), data)
    return data


# ---------------------------
# UPDATE
# ---------------------------
@router.put("/{collection_id}")
async def update_collection(collection_id: uuid.UUID, body: CollectionUpdate, key_info: RequireAdmin):
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Construire la requête dynamiquement (seulement les champs fournis)
            updates = []
            params = []
            if body.name is not None:
                updates.append("name = $1")
                params.append(body.name)
            if body.description is not None:
                updates.append("description = $" + str(len(params)+1))
                params.append(body.description)
            if body.config is not None:
                updates.append("config = $" + str(len(params)+1))
                params.append(body.config.model_dump())
            if body.access_level is not None:
                updates.append("access_level = $" + str(len(params)+1))
                params.append(body.access_level)
            updates.append("updated_at = NOW()")

            if not updates:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "No fields to update")

            query = f"""
                UPDATE collections
                SET {', '.join(updates)}
                WHERE id = ${len(params)+1}
                RETURNING *
            """
            params.append(collection_id)
            row = await conn.fetchrow(query, *params)

    if not row:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Collection not found")

    invalidate_collection_cache(str(collection_id))
    # Invalider le cache local aussi
    _collection_cache.pop(str(collection_id), None)
    return dict(row)


# ---------------------------
# DELETE (LOG SAFE)
# ---------------------------
@router.delete("/{collection_id}")
async def delete_collection(collection_id: uuid.UUID, key_info: RequireAdmin):
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM collections WHERE id = $1",
            collection_id
        )

    if result == "DELETE 0":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Collection not found")

    invalidate_collection_cache(str(collection_id))
    _collection_cache.pop(str(collection_id), None)
    return {"message": "Collection deleted"}


# ---------------------------
# STATS (OPTIMIZED)
# ---------------------------
@router.get("/{collection_id}/stats", response_model=CollectionStats)
async def get_stats(collection_id: uuid.UUID, key_info: RequireRead):
    pool = await get_pool()
    async with pool.acquire() as conn:
        stats = await conn.fetchrow(
            """
            SELECT
                c.id as collection_id,
                COUNT(DISTINCT d.id) as document_count,
                COALESCE(SUM(d.chunk_count), 0) as chunk_count,
                COALESCE(SUM(d.total_tokens), 0) as total_tokens,
                MAX(d.created_at) as last_ingested_at,
                c.created_at
            FROM collections c
            LEFT JOIN documents d ON d.collection_id = c.id
            WHERE c.id = $1
            GROUP BY c.id, c.created_at
            """,
            collection_id,
        )

    if not stats:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Collection not found")

    return dict(stats)


# ---------------------------
# CONFIG UPDATE
# ---------------------------
@router.put("/{collection_id}/config")
async def update_config(collection_id: uuid.UUID, config: CollectionConfig, key_info: RequireAdmin):
    if config.chunk_size > 4096:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "chunk_size too large (max 4096)")

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """UPDATE collections
               SET config = $1, updated_at = NOW()
               WHERE id = $2
               RETURNING *""",
            config.model_dump(),
            collection_id,
        )

    if not row:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Collection not found")

    invalidate_collection_cache(str(collection_id))
    _collection_cache.pop(str(collection_id), None)
    return dict(row)