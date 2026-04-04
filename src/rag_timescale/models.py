from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


def encode_cursor(created_at: datetime, id: uuid.UUID) -> str:
    payload = [created_at.isoformat(), str(id)]
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()


def decode_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
        return datetime.fromisoformat(payload[0]), uuid.UUID(payload[1])
    except (ValueError, IndexError, json.JSONDecodeError):
        raise ValueError("Invalid cursor format") from None


class AccessLevel(str, Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class ChunkStrategy(str, Enum):
    HIERARCHICAL = "hierarchical"
    SECTION = "section"
    SEMANTIC = "semantic"
    FIXED = "fixed"


class CollectionConfig(BaseModel):
    chunk_strategy: ChunkStrategy = ChunkStrategy.HIERARCHICAL
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    bm25_language: str = "auto"
    reranker_top_k: int = 10
    search_top_k: int = 20
    rrf_k: int = 60
    bm25_weight: float = 0.5
    vector_weight: float = 0.5
    metadata_schema: dict[str, Any] = Field(default_factory=dict)


class APIKeyCreate(BaseModel):
    name: str
    expires_at: datetime | None = None
    permissions: dict[str, list[AccessLevel]] = Field(
        default_factory=lambda: {
            "collections": ["*"],
            "roles": [AccessLevel.READ, AccessLevel.WRITE, AccessLevel.ADMIN],
        }  # type: ignore[arg-type]
    )


class APIKeyResponse(BaseModel):
    id: uuid.UUID
    name: str
    key: str
    created_at: datetime
    expires_at: datetime | None
    last_used_at: datetime | None


class CollectionCreate(BaseModel):
    name: str
    description: str = ""
    config: CollectionConfig = Field(default_factory=CollectionConfig)
    access_level: str = "private"


class CollectionResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: str
    config: CollectionConfig
    access_level: str
    owner_key_id: uuid.UUID
    created_at: datetime


class DocumentResponse(BaseModel):
    id: uuid.UUID
    collection_id: uuid.UUID
    filename: str
    mime_type: str
    size_bytes: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_count: int = 0
    created_at: datetime


class ChunkResponse(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    parent_id: uuid.UUID | None
    content: str
    section_path: list[str] = Field(default_factory=list)
    chunk_level: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None
    section_path: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    rerank: bool = True
    rerank_top_k: int = 10
    include_parent_context: bool = True
    parent_context_levels: int = 1
    filters: dict[str, Any] = Field(default_factory=dict)
    bm25_weight: float | None = None
    vector_weight: float | None = None
    diskann_search_list: int | None = None   # par défaut 200 
    diskann_rescore: int | None = None       # par défaut 100
    

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_chunks_searched: int
    elapsed_ms: float


class IngestResponse(BaseModel):
    document_id: uuid.UUID
    collection_id: uuid.UUID
    filename: str
    chunk_count: int
    created_at: datetime


class CollectionAccessGrant(BaseModel):
    api_key_id: uuid.UUID
    access_level: AccessLevel = AccessLevel.READ


class CollectionStats(BaseModel):
    collection_id: uuid.UUID
    document_count: int
    chunk_count: int
    total_tokens: int
    last_ingested_at: datetime | None
    created_at: datetime


class PaginatedResponse(BaseModel, Generic[T]):
    data: list[T]
    next_cursor: str | None = None
