from __future__ import annotations

import time
from typing import Any

from structlog import get_logger

from rag_timescale.auth.keys import get_key_by_hash
from rag_timescale.config import settings

log = get_logger()


class PermissionCache:
    def __init__(self, max_size: int = 1024, ttl: int = 60):
        self._cache: dict[str, tuple[Any, float]] = {}
        self._max_size = max_size
        self._ttl = ttl

    def get(self, key: str) -> Any | None:
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        self._cache[key] = (value, time.time())

    def invalidate(self, key: str) -> None:
        self._cache.pop(key, None)

    def invalidate_collection(self, collection_id: str) -> None:
        keys_to_remove = [k for k in self._cache if collection_id in k]
        for k in keys_to_remove:
            del self._cache[k]


_permission_cache = PermissionCache(
    max_size=settings.auth.permission_cache_max_size,
    ttl=settings.auth.permission_cache_ttl,
)


async def authenticate_key(raw_key: str) -> dict | None:
    cache_key = f"auth:{raw_key[:20]}"
    cached = _permission_cache.get(cache_key)
    if cached is not None:
        return cached

    key_info = await get_key_by_hash(raw_key)
    if key_info:
        _permission_cache.set(cache_key, key_info)
    return key_info


async def check_access_cached(
    key_id: str,
    collection_id: str,
    required_level: str,
    check_func,
) -> bool:
    cache_key = f"access:{key_id}:{collection_id}:{required_level}"
    cached = _permission_cache.get(cache_key)
    if cached is not None:
        return cached

    result = await check_func(key_id, collection_id, required_level)
    _permission_cache.set(cache_key, result)
    return result


def invalidate_access_cache(key: str) -> None:
    _permission_cache.invalidate(key)


def invalidate_collection_cache(collection_id: str) -> None:
    _permission_cache.invalidate_collection(collection_id)
