from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from rag_timescale.auth.keys import check_collection_access
from rag_timescale.auth.permissions import authenticate_key
from rag_timescale.models import AccessLevel

_bearer = HTTPBearer(auto_error=False)


async def _extract_key(request: Request, bearer: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> str | None:
    if bearer and bearer.credentials:
        return bearer.credentials
    api_key = request.headers.get("X-API-Key", "")
    return api_key if api_key else None


async def _get_optional_key(creds: str | None = Depends(_extract_key)) -> dict | None:
    if not creds:
        return None
    return await authenticate_key(creds)


async def _get_required_key(creds: str | None = Depends(_extract_key)) -> dict:
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


async def _check_access(collection_id: uuid.UUID, key_info: dict, level: AccessLevel) -> dict:
    has_access = await check_collection_access(
        key_id=key_info["id"],
        collection_id=collection_id,
        required_level=level,
    )
    if not has_access:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient permissions. Required: {level.value}",
        )
    return key_info


async def get_read_access(collection_id: uuid.UUID, key_info: dict = Depends(_get_required_key)) -> dict:
    return await _check_access(collection_id, key_info, AccessLevel.READ)


async def get_write_access(collection_id: uuid.UUID, key_info: dict = Depends(_get_required_key)) -> dict:
    return await _check_access(collection_id, key_info, AccessLevel.WRITE)


async def get_admin_access(collection_id: uuid.UUID, key_info: dict = Depends(_get_required_key)) -> dict:
    return await _check_access(collection_id, key_info, AccessLevel.ADMIN)


async def verify_key_ownership(key_id: uuid.UUID, key_info: dict = Depends(_get_required_key)) -> dict:
    if key_info["id"] != key_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only manage your own API keys",
        )
    return key_info


OptionalKey = Annotated[dict | None, Depends(_get_optional_key)]
RequireKey = Annotated[dict, Depends(_get_required_key)]
RequireRead = Annotated[dict, Depends(get_read_access)]
RequireWrite = Annotated[dict, Depends(get_write_access)]
RequireAdmin = Annotated[dict, Depends(get_admin_access)]
