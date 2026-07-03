# Industrial RAG API

Enterprise RAG (Retrieval-Augmented Generation) system with TimescaleDB, hybrid search, and intelligent chunking.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Authentication](#authentication)
- [Collections](#collections)
- [Documents](#documents)
- [Search](#search)
- [Keys & Permissions](#keys--permissions)
- [Cursor Pagination](#cursor-pagination)
- [API Reference](#api-reference)
- [Testing](#testing)

## Features

- **Hybrid Search**: Combines BM25 full-text search with vector similarity using Reciprocal Rank Fusion (RRF)
- **Intelligent Chunking**: Multiple strategies (hierarchical, section-based, semantic, fixed-size)
- **Document Parsing**: Supports PDF, DOCX, HTML, and plain text
- **Reranking**: Cross-encoder reranking for improved relevance
- **API Key Authentication**: Argon2-hashed keys with role-based access control
- **Cursor Pagination**: Efficient pagination for large datasets

## Prerequisites

- Python 3.10+
- PostgreSQL 15+ with TimescaleDB extension
- Docker (optional, for PostgreSQL setup)

## Installation

### 1. Clone and Install Dependencies

```bash
# Clone the repository
git clone <your-repo-url>
cd rag_timescale

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### 2. Set Up PostgreSQL with TimescaleDB

Using Docker:

```bash
# Start PostgreSQL with TimescaleDB
docker run -d \
  --name rag_timescale_db \
  -e POSTGRES_DB=rag \
  -e POSTGRES_USER=rag \
  -e POSTGRES_PASSWORD=rag_password \
  -p 5432:5432 \
  timescale/timescaledb:latest-pg15
```

Or use the provided `docker-compose.yml`:

```bash
docker-compose up -d
```

## Configuration

Create a `.env` file in the project root:

```env
# Database
DB_HOST=localhost
DB_PORT=5432
DB_USER=rag
DB_PASSWORD=rag_password
DB_NAME=rag

# Embedding Model
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=64

# Reranker Model
RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_DEVICE=cpu

# Authentication
AUTH_KEY_PREFIX=rag_
AUTH_KEY_LENGTH=48
AUTH_HASH_TIME_COST=3
AUTH_HASH_MEMORY_COST=65536
AUTH_HASH_PARALLELISM=4

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false
```

## Running the Application

```bash
# Start the API server
python -m uvicorn rag_timescale.main:app --reload --host 0.0.0.0 --port 8000
```

The API will:
1. Run database migrations automatically
2. Initialize the embedding model
3. Start serving at `http://localhost:8000`

API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## API Documentation

### Authentication

All API endpoints (except key creation) require authentication using an API key.

#### Using Headers

```bash
# X-API-Key header
curl -H "X-API-Key: your_api_key_here" http://localhost:8000/api/v1/collections

# Authorization header
curl -H "Authorization: Bearer your_api_key_here" http://localhost:8000/api/v1/collections
```

## Getting Started

### Step 1: Create an API Key

```bash
curl -X POST http://localhost:8000/api/v1/keys \
  -H "Content-Type: application/json" \
  -d '{"name": "My First Key"}'
```

Response:
```json
{
  "name": "My First Key",
  "key": "rag_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Important**: Save the `key` value - it's only shown once!

### Step 2: Create a Collection

```bash
curl -X POST http://localhost:8000/api/v1/collections \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-documents",
    "description": "My document collection"
  }'
```

Response:
```json
{
  "id": "660e8400-e29b-41d4-a716-446655440001",
  "name": "my-documents",
  "description": "My document collection",
  "config": {...},
  "access_level": "private",
  "owner_key_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2024-01-01T00:00:00Z"
}
```

### Step 3: Ingest a Document

```bash
curl -X POST http://localhost:8000/api/v1/collections/{collection_id}/ingest \
  -H "X-API-Key: your_api_key" \
  -F "file=@/path/to/document.pdf" \
  -F "title=My Document"
```

For batch ingestion:

```bash
curl -X POST http://localhost:8000/api/v1/collections/{collection_id}/batch/ingest \
  -H "X-API-Key: your_api_key" \
  -F "files=@/path/to/doc1.pdf" \
  -F "files=@/path/to/doc2.pdf" \
  -F "files=@/path/to/doc3.pdf"
```

### Step 4: Search Documents

```bash
curl -X POST http://localhost:8000/api/v1/collections/{collection_id}/search \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "top_k": 10,
    "rerank": true,
    "rerank_top_k": 5
  }'
```

Response:
```json
{
  "query": "What is machine learning?",
  "results": [
    {
      "chunk_id": "770e8400-e29b-41d4-a716-446655440002",
      "document_id": "660e8400-e29b-41d4-a716-446655440001",
      "content": "Machine learning is a subset of artificial intelligence...",
      "score": 0.95,
      "vector_rank": 1,
      "section_path": ["Chapter 1", "Introduction"]
    }
  ],
  "total_chunks_searched": 150,
  "elapsed_ms": 45.2
}
```

## Collections

Collections organize documents and define their processing configuration.

### Create Collection

```bash
curl -X POST http://localhost:8000/api/v1/collections \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "technical-docs",
    "description": "Technical documentation",
    "config": {
      "chunk_strategy": "hierarchical",
      "chunk_size": 512,
      "chunk_overlap": 64,
      "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
      "embedding_dimensions": 384
    }
  }'
```

### List Collections (with Pagination)

```bash
# Get first page
curl "http://localhost:8000/api/v1/collections?limit=10" \
  -H "X-API-Key: your_api_key"

# Response
{
  "data": [...],
  "next_cursor": "eyJjcml0ZWRfYXQiOiIyMDI0LTAxLTAxVDAwOjAwOjAwKzAwOjAwIiwiZGJfaWQiOiI2NjBlODQwMC1lMjliLTQxZDQtcTcxNi00NDY2NTU0NDAwMDEifQ=="
}
```

### Update Collection Config

```bash
curl -X PUT http://localhost:8000/api/v1/collections/{collection_id}/config \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "chunk_strategy": "semantic",
    "chunk_size": 256
  }'
```

## Documents

### Document Fields

Documents have the following fields:
- `id`: Unique identifier (UUID)
- `collection_id`: Parent collection ID
- `external_id`: Optional external reference
- `title`: Document title
- `filename`: Original filename
- `mime_type`: File type
- `size_bytes`: File size
- `chunk_count`: Number of chunks
- `total_tokens`: Total token count
- `metadata`: Custom metadata
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp

### Ingest Document

```bash
# Single document
curl -X POST http://localhost:8000/api/v1/collections/{collection_id}/ingest \
  -H "X-API-Key: your_api_key" \
  -F "file=@document.pdf" \
  -F "title=My PDF Document" \
  -F "external_id=doc-001"

# With metadata
curl -X POST http://localhost:8000/api/v1/collections/{collection_id}/ingest \
  -H "X-API-Key: your_api_key" \
  -F "file=@document.pdf" \
  -F "metadata={\"author\": \"John Doe\", \"category\": \"technical\"}"
```

### Update Document

Replace document content while keeping the same ID:

```bash
curl -X POST http://localhost:8000/api/v1/collections/{collection_id}/documents/{document_id}/update \
  -H "X-API-Key: your_api_key" \
  -F "file=@updated-document.pdf"
```

### List Documents (with Pagination)

```bash
# First page
curl "http://localhost:8000/api/v1/collections/{collection_id}/documents?limit=20" \
  -H "X-API-Key: your_api_key"

# Get next page using cursor
curl "http://localhost:8000/api/v1/collections/{collection_id}/documents?limit=20&cursor=eyJjcml0ZWRfYXQiOi..." \
  -H "X-API-Key: your_api_key"
```

### Delete Document

```bash
# By ID
curl -X DELETE http://localhost:8000/api/v1/collections/{collection_id}/documents/{document_id} \
  -H "X-API-Key: your_api_key"

# By external ID
curl -X DELETE http://localhost:8000/api/v1/collections/{collection_id}/documents/by-external-id/doc-001 \
  -H "X-API-Key: your_api_key"
```

### Get Document

```bash
# Basic
curl "http://localhost:8000/api/v1/collections/{collection_id}/documents/{document_id}" \
  -H "X-API-Key: your_api_key"

# With chunks
curl "http://localhost:8000/api/v1/collections/{collection_id}/documents/{document_id}?include_chunks=true" \
  -H "X-API-Key: your_api_key"

# By external ID
curl "http://localhost:8000/api/v1/collections/{collection_id}/documents/by-external-id/doc-001" \
  -H "X-API-Key: your_api_key"
```

## Search

### Basic Search

```bash
curl -X POST http://localhost:8000/api/v1/collections/{collection_id}/search \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "your search question"}'
```

### Advanced Search Options

```bash
curl -X POST http://localhost:8000/api/v1/collections/{collection_id}/search \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "your search question",
    "top_k": 20,
    "rerank": true,
    "rerank_top_k": 5,
    "include_parent_context": true,
    "parent_context_levels": 2,
    "filters": {"chunk_level": 1},
    "bm25_weight": 0.3,
    "vector_weight": 0.7
  }'
```

### Search Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Search query text |
| `top_k` | int | 10 | Number of final results |
| `rerank` | bool | true | Enable cross-encoder reranking |
| `rerank_top_k` | int | 10 | Results to rerank |
| `include_parent_context` | bool | true | Include parent chunks |
| `parent_context_levels` | int | 1 | Parent chunk levels to include |
| `filters` | object | {} | Metadata filters |
| `bm25_weight` | float | 0.5 | BM25 contribution weight |
| `vector_weight` | float | 0.5 | Vector search contribution weight |
| `diskann_search_list` | int | 200 | DiskANN search list size |
| `diskann_rescore` | int | 100 | DiskANN rescore count |

### Keyword Search (BM25 Only)

```bash
curl -X POST http://localhost:8000/api/v1/collections/{collection_id}/search/bm25 \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "exact keywords", "limit": 10}'
```

### Vector Search (Embedding Similarity)

```bash
curl -X POST http://localhost:8000/api/v1/collections/{collection_id}/search/vector \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "semantic meaning", "limit": 10}'
```

### Fast Raw Search

```bash
curl -X POST http://localhost:8000/api/v1/collections/{collection_id}/search/raw \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "fast search without reranking"}'
```

## Keys & Permissions

### API Key Endpoints

#### Create Key

```bash
curl -X POST http://localhost:8000/api/v1/keys \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Production Key",
    "permissions": {
      "collections": ["*"],
      "roles": ["read", "write", "admin"]
    }
  }'
```

#### List Your Keys

```bash
curl http://localhost:8000/api/v1/keys \
  -H "X-API-Key: your_api_key"
```

#### Revoke Key

```bash
curl -X DELETE http://localhost:8000/api/v1/keys/{key_id} \
  -H "X-API-Key: your_api_key"
```

### Collection Access Control

Grant other keys access to your collection:

```bash
curl -X POST http://localhost:8000/api/v1/collections/{collection_id}/keys \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "api_key_id": "other-key-id",
    "access_level": "read"
  }'
```

### List Collection Keys (with Pagination)

```bash
curl "http://localhost:8000/api/v1/collections/{collection_id}/keys?limit=10" \
  -H "X-API-Key: your_api_key"
```

### Access Levels

| Level | Permissions |
|-------|-------------|
| `read` | View collections, documents, search |
| `write` | read + ingest documents |
| `admin` | write + manage collection, delete documents, manage keys |

### Remove Key Access

```bash
curl -X DELETE http://localhost:8000/api/v1/collections/{collection_id}/keys/{api_key_id} \
  -H "X-API-Key: your_api_key"
```

## Cursor Pagination

All list endpoints support cursor-based pagination for efficient traversal of large datasets.

### How It Works

1. Make a request without a cursor to get the first page
2. If there are more results, `next_cursor` is included in the response
3. Use `next_cursor` in the next request to get subsequent pages
4. When `next_cursor` is `null`, you've reached the end

### Pagination Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 20 | Items per page (1-100) |
| `cursor` | string | null | Opaque cursor from previous response |

### Example: Paginating Through Collections

```bash
# Page 1
curl "http://localhost:8000/api/v1/collections?limit=5" \
  -H "X-API-Key: your_api_key"

# Response
{
  "data": [
    {"id": "...", "name": "Collection 1", ...},
    {"id": "...", "name": "Collection 2", ...},
    {"id": "...", "name": "Collection 3", ...},
    {"id": "...", "name": "Collection 4", ...},
    {"id": "...", "name": "Collection 5", ...}
  ],
  "next_cursor": "eyJjcml0ZWRfYXQiOiIyMDI0LTAxLTAxVDAwOjAwOjAwWiIsImlkIjoiLi4uIn0="
}

# Page 2
curl "http://localhost:8000/api/v1/collections?limit=5&cursor=eyJjcml0ZWRfYXQiOi4uLn0=" \
  -H "X-API-Key: your_api_key"

# Page 3 (last page - no next_cursor)
{
  "data": [...],
  "next_cursor": null
}
```

### Why Cursor Pagination?

1. **Stable**: Cursors don't shift when data is inserted/deleted
2. **Efficient**: Uses keyset pagination, faster than OFFSET
3. **Scalable**: Works well with millions of rows

## Chunking Strategies

Configure how documents are split into chunks:

### Hierarchical (Default)

Best for structured documents with headings:

```json
{
  "chunk_strategy": "hierarchical",
  "chunk_size": 512,
  "chunk_overlap": 64
}
```

### Section-Based

Splits by detected sections:

```json
{
  "chunk_strategy": "section"
}
```

### Semantic

Groups semantically similar content:

```json
{
  "chunk_strategy": "semantic",
  "chunk_size": 256
}
```

### Fixed-Size

Simple character/word-based splitting:

```json
{
  "chunk_strategy": "fixed",
  "chunk_size": 1000
}
```

## API Reference

### Health Check

```bash
curl http://localhost:8000/health
# Response: {"status": "ok"}
```

### Full API Routes

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/v1/keys` | - | Create API key |
| GET | `/api/v1/keys` | Key | List your keys |
| DELETE | `/api/v1/keys/{id}` | Key (owner) | Revoke key |
| POST | `/api/v1/collections` | Key | Create collection |
| GET | `/api/v1/collections` | Key | List collections (paginated) |
| GET | `/api/v1/collections/{id}` | Read | Get collection |
| PUT | `/api/v1/collections/{id}` | Admin | Update collection |
| DELETE | `/api/v1/collections/{id}` | Admin | Delete collection |
| GET | `/api/v1/collections/{id}/stats` | Read | Collection statistics |
| PUT | `/api/v1/collections/{id}/config` | Admin | Update config |
| POST | `/api/v1/collections/{id}/ingest` | Write | Ingest document |
| POST | `/api/v1/collections/{id}/batch/ingest` | Write | Batch ingest |
| GET | `/api/v1/collections/{id}/documents` | Read | List documents (paginated) |
| GET | `/api/v1/collections/{id}/documents/{id}` | Read | Get document |
| GET | `/api/v1/collections/{id}/documents/by-external-id/{id}` | Read | Get by external ID |
| POST | `/api/v1/collections/{id}/documents/{id}/update` | Write | Update document |
| DELETE | `/api/v1/collections/{id}/documents/{id}` | Admin | Delete document |
| DELETE | `/api/v1/collections/{id}/documents/by-external-id/{id}` | Admin | Delete by external ID |
| GET | `/api/v1/collections/{id}/chunks/{id}` | Read | Get chunk |
| POST | `/api/v1/collections/{id}/search` | Read | Hybrid search |
| POST | `/api/v1/collections/{id}/search/bm25` | Read | BM25 search |
| POST | `/api/v1/collections/{id}/search/vector` | Read | Vector search |
| POST | `/api/v1/collections/{id}/search/raw` | Read | Fast raw search |
| POST | `/api/v1/collections/{id}/keys` | Admin | Grant access |
| GET | `/api/v1/collections/{id}/keys` | Admin | List keys (paginated) |
| DELETE | `/api/v1/collections/{id}/keys/{id}` | Admin | Revoke access |

## Testing

### Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_collections_pagination.py -v

# Run with coverage
python -m pytest tests/ --cov=src/rag_timescale --cov-report=html
```

### Test Structure

```
tests/
├── conftest.py                    # Shared fixtures
├── test_collections_pagination.py # Collection list tests
├── test_documents_pagination.py   # Document list tests
├── test_permissions_pagination.py # Permission list tests
├── test_search_api.py            # Search endpoint tests
├── test_keys_api.py              # API keys tests
└── test_chunking.py              # Chunking strategy tests
```

## Troubleshooting

### Database Connection Issues

```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Test connection
psql -h localhost -U rag -d rag -c "SELECT 1;"
```

### Embedding Model Issues

The first search may take longer as the model downloads. Check logs:

```bash
python -m uvicorn rag_timescale.main:app --reload --log-level debug
```

### Permission Denied Errors

1. Ensure you're using a valid API key
2. Check the key has the required access level
3. Verify the collection exists and is accessible

## Development

### Project Structure

```
rag_timescale/
├── src/rag_timescale/
│   ├── api/              # FastAPI routes
│   ├── auth/             # Authentication
│   ├── chunking/         # Document chunking
│   ├── db/               # Database
│   ├── embeddings/       # Embedding provider
│   ├── parsers/         # Document parsers
│   ├── reranker/        # Cross-encoder
│   └── retrieval/        # Search retrieval
├── tests/               # Test suite
└── docker-compose.yml    # Docker setup
```

### Code Quality

```bash
# Lint
ruff check src/

# Format
ruff format src/
```

## License

MIT License
