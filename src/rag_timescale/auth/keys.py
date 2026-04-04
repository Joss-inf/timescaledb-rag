from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timezone

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from structlog import get_logger

from rag_timescale.config import settings
from rag_timescale.db.connection import get_pool
from rag_timescale.models import AccessLevel

log = get_logger()
_ph = PasswordHasher(
    time_cost=settings.auth.hash_time_cost,
    memory_cost=settings.auth.hash_memory_cost,
    parallelism=settings.auth.hash_parallelism,
)


def generate_api_key() -> tuple[str, str]:
    raw_key = f"{settings.auth.key_prefix}{secrets.token_urlsafe(settings.auth.key_length)}"
    key_hash = _ph.hash(raw_key)
    return raw_key, key_hash


def verify_api_key(raw_key: str, stored_hash: str) -> bool:
    try:
        _ph.verify(stored_hash, raw_key)
        return True
    except VerifyMismatchError:
        return False


async def create_api_key(
    name: str,
    permissions: dict,
    expires_at: datetime | None = None,
) -> tuple[str, dict]:
    raw_key, key_hash = generate_api_key()
    pool = await get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO api_keys (key_hash, name, permissions, expires_at)
            VALUES ($1, $2, $3, $4)
            RETURNING id, name, created_at, expires_at
            """,
            key_hash,
            name,
            permissions,
            expires_at,
        )

    log.info("api_key_created", key_id=str(row["id"]), name=name)
    return raw_key, {
        "id": row["id"],
        "name": row["name"],
        "created_at": row["created_at"],
        "expires_at": row["expires_at"],
    }


async def revoke_api_key(key_id: uuid.UUID) -> bool:
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE api_keys SET is_active = FALSE WHERE id = $1 AND is_active = TRUE
            """,
            key_id,
        )
    revoked = result == "UPDATE 1"
    if revoked:
        log.info("api_key_revoked", key_id=str(key_id))
    return revoked


async def list_api_keys(owner_key_id: uuid.UUID) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, name, created_at, expires_at, last_used_at, is_active
            FROM api_keys
            WHERE id = $1 
               OR id IN (SELECT api_key_id FROM collection_access WHERE collection_id IN 
                            (SELECT id FROM collections WHERE owner_key_id = $1))
            ORDER BY created_at DESC
            """,
            owner_key_id,
        )
    return [dict(r) for r in rows]


async def get_key_by_hash(raw_key: str) -> dict | None:
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, key_hash, name, permissions, expires_at FROM api_keys WHERE is_active = TRUE"
        )

        target_row = None
     
        for row in rows:
            if verify_api_key(raw_key, row["key_hash"]):
                target_row = row
                break
        
        if not target_row:
            return None

        if target_row["expires_at"] and target_row["expires_at"] < datetime.now(timezone.utc):
            log.info("api_key_expired", key_id=str(target_row["id"]))
            # Optionnel : on pourrait désactiver la clé automatiquement ici
            return None

        await conn.execute(
            "UPDATE api_keys SET last_used_at = NOW() WHERE id = $1",
            target_row["id"],
        )

        return {
            "id": target_row["id"],
            "name": target_row["name"],
            "permissions": target_row["permissions"],
            "expires_at": target_row["expires_at"],
        }


async def check_collection_access(
    key_id: uuid.UUID,
    collection_id: uuid.UUID,
    required_level: AccessLevel = AccessLevel.READ,
) -> bool:
    level_order = {AccessLevel.READ: 0, AccessLevel.WRITE: 1, AccessLevel.ADMIN: 2}
    required = level_order[required_level]

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                CASE
                    WHEN c.owner_key_id = $1 THEN 'admin'
                    WHEN ca.access_level IS NOT NULL THEN ca.access_level
                    WHEN c.access_level = 'public' THEN 'read'
                    ELSE NULL
                END as effective_level
            FROM collections c
            LEFT JOIN collection_access ca ON ca.collection_id = c.id AND ca.api_key_id = $1
            WHERE c.id = $2
            """,
            key_id,
            collection_id,
        )

    if not row or not row["effective_level"]:
        return False

    effective = level_order.get(AccessLevel(row["effective_level"]), -1)
    return effective >= required
