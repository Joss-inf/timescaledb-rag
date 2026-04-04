from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, status

from rag_timescale.api.deps import RequireKey
from rag_timescale.auth.keys import create_api_key, revoke_api_key
from rag_timescale.models import APIKeyCreate, APIKeyResponse

router = APIRouter()

@router.post("/", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_key(body: APIKeyCreate):
    raw_key, key_data = await create_api_key(
        name=body.name,
        permissions=body.permissions,
        expires_at=body.expires_at,
    )
    return APIKeyResponse(
        id=key_data["id"],
        name=key_data["name"],
        key=raw_key,
        created_at=key_data["created_at"],
        expires_at=key_data["expires_at"],
        last_used_at=None,
    )


@router.delete("/{key_id}")
async def delete_key(key_id: uuid.UUID, key_info: RequireKey):
    if key_info["id"] != key_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only revoke your own API key",
        )
    success = await revoke_api_key(key_id)
    if not success:
        raise HTTPException(status_code=404, detail="API key not found or already revoked")
    return {"message": "API key revoked"}
