from __future__ import annotations

from structlog import get_logger

from rag_timescale.db.connection import get_pool

log = get_logger()

MIGRATIONS = [
    """
    CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
    """,
    """
    CREATE EXTENSION IF NOT EXISTS pg_textsearch;
    """,
    """
    CREATE TABLE IF NOT EXISTS api_keys (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        key_hash TEXT NOT NULL,
        name TEXT NOT NULL,
        permissions JSONB NOT NULL DEFAULT '{"collections": ["*"], "roles": ["read", "write", "admin"]}'::jsonb,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        expires_at TIMESTAMPTZ,
        last_used_at TIMESTAMPTZ,
        is_active BOOLEAN DEFAULT TRUE
    );

    CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);
    CREATE INDEX IF NOT EXISTS idx_api_keys_is_active ON api_keys(is_active) WHERE is_active = TRUE;
    """,
    """
    CREATE TABLE IF NOT EXISTS collections (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        name TEXT NOT NULL,
        description TEXT NOT NULL DEFAULT '',
        config JSONB NOT NULL DEFAULT '{}'::jsonb,
        access_level TEXT NOT NULL DEFAULT 'private',
        owner_key_id UUID NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE UNIQUE INDEX IF NOT EXISTS idx_collections_name_owner ON collections(name, owner_key_id);
    CREATE INDEX IF NOT EXISTS idx_collections_access_level ON collections(access_level);
    """,
    """
    CREATE TABLE IF NOT EXISTS collection_access (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
        api_key_id UUID NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
        access_level TEXT NOT NULL DEFAULT 'read',
        granted_at TIMESTAMPTZ DEFAULT NOW(),
        granted_by UUID REFERENCES api_keys(id),
        UNIQUE(collection_id, api_key_id)
    );

    CREATE INDEX IF NOT EXISTS idx_collection_access_collection ON collection_access(collection_id);
    CREATE INDEX IF NOT EXISTS idx_collection_access_key ON collection_access(api_key_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS documents (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
        filename TEXT NOT NULL,
        mime_type TEXT NOT NULL,
        size_bytes BIGINT NOT NULL DEFAULT 0,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        chunk_count INT NOT NULL DEFAULT 0,
        total_tokens INT NOT NULL DEFAULT 0,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection_id);
    CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
    CREATE INDEX IF NOT EXISTS idx_documents_external_id
    ON documents(collection_id, external_id) WHERE external_id IS NOT NULL;
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
        document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        parent_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
        content TEXT NOT NULL,
        embedding vector(384),
        section_path TEXT[] NOT NULL DEFAULT '{}',
        chunk_level INT NOT NULL DEFAULT 0,
        token_count INT NOT NULL DEFAULT 0,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks(collection_id);
    CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
    CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);
    CREATE INDEX IF NOT EXISTS idx_chunks_level ON chunks(chunk_level);
    """,
    """
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS title TEXT;
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS external_id TEXT;
    """,
]


async def run_migrations(embedding_dimensions: int = 384) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        for i, migration in enumerate(MIGRATIONS):
            stmt = migration
            if i == 6:
                stmt = migration.replace("vector(384)", f"vector({embedding_dimensions})")
            await conn.execute(stmt)
            log.info("migration_applied", index=i)

        await _create_bm25_indexes(conn)
        await _create_diskann_index(conn, embedding_dimensions)
        await _create_sql_functions(conn)

    log.info("all_migrations_completed")


async def _create_bm25_indexes(conn) -> None:
    await conn.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_indexes WHERE indexname = 'idx_chunks_bm25'
            ) THEN
                CREATE INDEX idx_chunks_bm25 ON chunks USING bm25(content)
                WITH (text_config = 'english');
            END IF;
        END
        $$;
    """)
    log.info("bm25_index_ensured")


async def _create_diskann_index(conn, dimensions: int) -> None:
    await conn.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_indexes WHERE indexname = 'idx_chunks_vector'
            ) THEN
                CREATE INDEX idx_chunks_vector ON chunks
                USING diskann (embedding vector_cosine_ops);
            END IF;
        END
        $$;
    """)
    log.info("diskann_index_ensured")


async def _create_sql_functions(conn) -> None:
    await conn.execute("""
        CREATE OR REPLACE FUNCTION hybrid_search(
            p_collection_id UUID,
            p_query_text TEXT,
            p_query_embedding vector,
            p_bm25_limit INT DEFAULT 20,
            p_vector_limit INT DEFAULT 20,
            p_bm25_weight FLOAT DEFAULT 0.5,
            p_vector_weight FLOAT DEFAULT 0.5,
            p_rrf_k INT DEFAULT 60,
            p_match_count INT DEFAULT 10,
            p_filters JSONB DEFAULT '{}'::jsonb
        )
        RETURNS TABLE (
            id UUID,
            document_id UUID,
            content TEXT,
            section_path TEXT[],
            chunk_level INT,
            metadata JSONB,
            bm25_rank INT,
            vector_rank INT,
            rrf_score FLOAT
        )
        AS $$
        DECLARE
            v_bm25_query TEXT;
        BEGIN
            v_bm25_query := p_query_text;

            RETURN QUERY
            WITH
            bm25_results AS (
                SELECT
                    c.id,
                    ROW_NUMBER() OVER (
                        ORDER BY c.content <@> to_bm25query(v_bm25_query, 'idx_chunks_bm25')
                    ) as rank
                FROM chunks c
                WHERE c.collection_id = p_collection_id
                ORDER BY c.content <@> to_bm25query(v_bm25_query, 'idx_chunks_bm25')
                LIMIT p_bm25_limit
            ),
            vector_results AS (
                SELECT
                    c.id,
                    ROW_NUMBER() OVER (
                        ORDER BY c.embedding <=> p_query_embedding
                    ) as rank
                FROM chunks c
                WHERE c.collection_id = p_collection_id
                ORDER BY c.embedding <=> p_query_embedding
                LIMIT p_vector_limit
            )
            SELECT
                c.id,
                c.document_id,
                c.content,
                c.section_path,
                c.chunk_level,
                c.metadata,
                b.rank,
                v.rank,
                (p_bm25_weight * COALESCE(1.0 / (p_rrf_k + b.rank), 0)) +
                (p_vector_weight * COALESCE(1.0 / (p_rrf_k + v.rank), 0)) as rrf_score
            FROM chunks c
            LEFT JOIN bm25_results b ON c.id = b.id
            LEFT JOIN vector_results v ON c.id = v.id
            WHERE b.id IS NOT NULL OR v.id IS NOT NULL
            ORDER BY rrf_score DESC
            LIMIT p_match_count;
        END;
        $$ LANGUAGE plpgsql;
    """)

    await conn.execute("""
        CREATE OR REPLACE FUNCTION get_parent_context(
            p_chunk_id UUID,
            p_levels INT DEFAULT 1
        )
        RETURNS TABLE (
            chunk_id UUID,
            content TEXT,
            section_path TEXT[],
            chunk_level INT
        )
        AS $$
        WITH RECURSIVE parent_chain AS (
            SELECT id, parent_id, content, section_path, chunk_level, 0 as depth
            FROM chunks
            WHERE id = p_chunk_id
            UNION ALL
            SELECT c.id, c.parent_id, c.content, c.section_path, c.chunk_level, pc.depth + 1
            FROM chunks c
            INNER JOIN parent_chain pc ON c.id = pc.parent_id
            WHERE pc.depth < p_levels
        )
        SELECT id as chunk_id, content, section_path, chunk_level
        FROM parent_chain
        WHERE id != p_chunk_id
        ORDER BY chunk_level ASC;
        $$ LANGUAGE SQL;
    """)

    log.info("sql_functions_created")
