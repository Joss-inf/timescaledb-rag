from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from rag_timescale.auth.keys import check_collection_access
from rag_timescale.auth.permissions import authenticate_key
from rag_timescale.models import AccessLevel

_bearer = HTTPBearer(auto_error=False)


async def _extract_key(
    request: Request,
    bearer: HTTPAuthorizationCredentials | None = Depends(_bearer)
) -> str | None:
    """Extrait la clé API du header Authorization (Bearer) ou X-API-Key."""
    if bearer and bearer.credentials:
        return bearer.credentials
    api_key = request.headers.get("X-API-Key")
    return api_key if api_key else None


async def _get_optional_key(creds: str | None = Depends(_extract_key)) -> dict | None:
    """Retourne les infos de la clé si fournie et valide, sinon None."""
    if not creds:
        return None
    return await authenticate_key(creds)


async def _get_required_key(creds: str | None = Depends(_extract_key)) -> dict:
    """Retourne les infos de la clé ou lève 401."""
    if not creds:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Use Authorization: Bearer <key> or X-API-Key header.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    key_info = await authenticate_key(creds)
    if not key_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key",
        )
    return key_info


# Cache simple pour les vérifications d'accès (TTL 5 secondes)
_access_cache: dict[tuple[str, str, str], tuple[float, bool]] = {}
CACHE_TTL = 5.0


async def _check_access(
    collection_id: uuid.UUID,
    key_info: dict,
    level: AccessLevel
) -> dict:
    """Vérifie l'accès avec cache court."""
    cache_key = (str(collection_id), str(key_info["id"]), level.value)
    import time
    now = time.time()
    cached = _access_cache.get(cache_key)
    if cached and now - cached[0] < CACHE_TTL:
        has_access = cached[1]
    else:
        has_access = await check_collection_access(
            key_id=key_info["id"],
            collection_id=collection_id,
            required_level=level,
        )
        _access_cache[cache_key] = (now, has_access)

    if not has_access:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient permissions. Required: {level.value}",
        )
    return key_info


def _get_access_dependency(level: AccessLevel):
    """Factory pour créer une dépendance d'accès paramétrée."""
    async def dependency(
        collection_id: uuid.UUID,
        key_info: dict = Depends(_get_required_key)
    ) -> dict:
        return await _check_access(collection_id, key_info, level)
    return dependency


# Exports des dépendances publiques
RequireRead = Annotated[dict, Depends(_get_access_dependency(AccessLevel.READ))]
RequireWrite = Annotated[dict, Depends(_get_access_dependency(AccessLevel.WRITE))]
RequireAdmin = Annotated[dict, Depends(_get_access_dependency(AccessLevel.ADMIN))]
OptionalKey = Annotated[dict | None, Depends(_get_optional_key)]
RequireKey = Annotated[dict, Depends(_get_required_key)]


async def verify_key_ownership(
    key_id: uuid.UUID,
    key_info: dict = Depends(_get_required_key)
) -> dict:
    """Vérifie que la clé authentifiée est bien celle demandée."""
    if key_info["id"] != key_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only manage your own API keys",
        )
    return key_info