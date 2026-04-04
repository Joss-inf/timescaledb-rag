from __future__ import annotations

import uuid
from structlog import get_logger
from rag_timescale.db.connection import get_pool

log = get_logger()

# Migration SQL : création des tables (sans les index lourds)
MIGRATIONS = [
    # 0️⃣ Extensions
    """
    CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
    CREATE EXTENSION IF NOT EXISTS pg_textsearch;
    """,

    # 1️⃣ API Keys
    """
    CREATE TABLE IF NOT EXISTS api_keys (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        key_hash TEXT NOT NULL,
        name TEXT NOT NULL,
        permissions JSONB NOT NULL DEFAULT '{"collections": ["*"], "roles": ["read", "write", "admin"]}',
        created_at TIMESTAMPTZ DEFAULT NOW(),
        expires_at TIMESTAMPTZ,
        last_used_at TIMESTAMPTZ,
        is_active BOOLEAN DEFAULT TRUE
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);
    CREATE INDEX IF NOT EXISTS idx_api_keys_is_active ON api_keys(is_active) WHERE is_active = TRUE;
    """,

    # 2️⃣ Collections
    """
    CREATE TABLE IF NOT EXISTS collections (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        name TEXT NOT NULL,
        description TEXT NOT NULL DEFAULT '',
        config JSONB NOT NULL DEFAULT '{}',
        access_level TEXT NOT NULL DEFAULT 'private',
        owner_key_id UUID NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
        collection_size_bytes BIGINT NOT NULL DEFAULT 0,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE UNIQUE INDEX idx_collections_name_owner ON collections(name, owner_key_id);
    CREATE INDEX idx_collections_access_level ON collections(access_level);
    CREATE INDEX idx_collections_owner ON collections(owner_key_id);
    CREATE INDEX idx_collections_pagination ON collections(created_at DESC, id DESC);
    """,

    # 3️⃣ Collection access
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
    CREATE INDEX idx_collection_access_collection ON collection_access(collection_id);
    CREATE INDEX idx_collection_access_key ON collection_access(api_key_id);
    CREATE INDEX idx_collection_access_lookup ON collection_access(collection_id, api_key_id);
    """,

    # 4️⃣ Documents (avec soft delete)
    """
    CREATE TABLE IF NOT EXISTS documents (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
        filename TEXT NOT NULL,
        mime_type TEXT NOT NULL,
        size_bytes BIGINT NOT NULL DEFAULT 0,
        metadata JSONB NOT NULL DEFAULT '{}',
        chunk_count INT NOT NULL DEFAULT 0,
        total_tokens INT NOT NULL DEFAULT 0,
        title TEXT,
        external_id TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW(),
        deleted_at TIMESTAMPTZ
    );
    CREATE INDEX idx_documents_collection ON documents(collection_id);
    CREATE INDEX idx_documents_filename ON documents(filename);
    CREATE INDEX idx_documents_external_id ON documents(collection_id, external_id) WHERE external_id IS NOT NULL;
    CREATE INDEX idx_documents_collection_stats ON documents(collection_id, created_at);
    CREATE INDEX idx_documents_active ON documents(collection_id) WHERE deleted_at IS NULL;
    """,

    # 5️⃣ Chunks (partitionnée par collection_id, avec soft delete)
    """
    CREATE TABLE IF NOT EXISTS chunks (
        id UUID NOT NULL,
        collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
        document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        parent_id UUID,
        content TEXT NOT NULL,
        embedding vector(384),
        section_path TEXT[] NOT NULL DEFAULT '{}',
        chunk_level INT NOT NULL DEFAULT 0,
        token_count INT NOT NULL DEFAULT 0,
        metadata JSONB NOT NULL DEFAULT '{}',
        created_at TIMESTAMPTZ DEFAULT NOW(),
        deleted_at TIMESTAMPTZ,
        PRIMARY KEY (collection_id, id)
    ) PARTITION BY LIST (collection_id);

    -- Index légers (créés immédiatement)
    CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks(collection_id);
    CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
    CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);
    CREATE INDEX IF NOT EXISTS idx_chunks_level ON chunks(chunk_level);
    CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON chunks USING GIN (metadata jsonb_path_ops);
    CREATE INDEX IF NOT EXISTS idx_chunks_metadata_col ON chunks USING GIN (collection_id, metadata jsonb_path_ops);
    CREATE INDEX IF NOT EXISTS idx_chunks_active ON chunks(collection_id, id) WHERE deleted_at IS NULL;

    -- Optimisations physiques
    ALTER TABLE chunks SET (fillfactor = 80);
    ALTER TABLE chunks SET (autovacuum_vacuum_scale_factor = 0.01);
    ALTER TABLE chunks SET (autovacuum_vacuum_threshold = 5000);
    ALTER TABLE chunks SET (autovacuum_analyze_scale_factor = 0.005);
    ALTER TABLE chunks SET (autovacuum_analyze_threshold = 2500);
    """,
]


async def run_migrations(embedding_dimensions: int = 384, create_indexes: bool = False) -> None:
    """
    Crée les tables sans les index lourds (BM25, DiskANN).
    Mettre create_indexes=True uniquement après le chargement initial des données.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        for i, migration in enumerate(MIGRATIONS):
            stmt = migration
            if i == 5:  # chunks table
                stmt = migration.replace("vector(384)", f"vector({embedding_dimensions})")
            await conn.execute(stmt)
            log.info("migration_applied", index=i)

        if create_indexes:
            await create_all_indexes(embedding_dimensions)

    log.info("all_migrations_completed")


async def create_all_indexes(embedding_dimensions: int = 384) -> None:
    """Crée les index BM25 et DiskANN (à exécuter après chargement initial)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Configuration des ressources pour les builds parallèles
        await conn.execute("SET maintenance_work_mem = '1GB'")
        await conn.execute("SET max_parallel_maintenance_workers = 4")

        await _create_bm25_index(conn)
        await _create_diskann_index(conn, embedding_dimensions)
        await _create_sql_functions(conn)
        await _configure_search_settings(conn)
        log.info("all_indexes_created")


async def _create_bm25_index(conn):
    await conn.execute("""
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_chunks_bm25') THEN
            CREATE INDEX CONCURRENTLY idx_chunks_bm25
            ON chunks USING bm25(content)
            WITH (text_config='english');
        END IF;
    END
    $$;
    """)
    log.info("bm25_index_created")


async def _create_diskann_index(conn, dimensions: int):
    await conn.execute(f"""
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_chunks_vector') THEN
            CREATE INDEX CONCURRENTLY idx_chunks_vector
            ON chunks USING diskann (embedding vector_cosine_ops)
            WITH (storage_layout = 'memory_optimized');
        END IF;
    END
    $$;
    """)
    log.info("diskann_index_created", dimensions=dimensions)


async def _configure_search_settings(conn):
    # Réglages par défaut pour les sessions (peuvent être modifiés par requête)
    await conn.execute("""
    ALTER DATABASE current_database() SET pg_textsearch.default_limit = 5000;
    ALTER DATABASE current_database() SET diskann.query_search_list_size = 200;
    ALTER DATABASE current_database() SET diskann.query_rescore = 100;
    """)
    log.info("search_settings_configured")


async def _create_sql_functions(conn):
    # hybrid_search avec prise en compte de deleted_at
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
        chunk_id UUID,
        document_id UUID,
        content TEXT,
        section_path TEXT[],
        chunk_level INT,
        metadata JSONB,
        bm25_rank INT,
        vector_rank INT,
        rrf_score FLOAT
    ) AS $$
    DECLARE
        bm25_idx TEXT := 'idx_chunks_bm25';
        bm25_q bm25query;
    BEGIN
        bm25_q := to_bm25query(p_query_text, bm25_idx);
        RETURN QUERY
        WITH filtered_chunks AS MATERIALIZED (
            SELECT id, document_id, content, section_path, chunk_level, metadata, embedding
            FROM chunks
            WHERE collection_id = p_collection_id
              AND deleted_at IS NULL
              AND (p_filters = '{}'::jsonb OR metadata @> p_filters)
        ),
        bm25_results AS MATERIALIZED (
            SELECT id AS chunk_id,
                   ROW_NUMBER() OVER (ORDER BY content <@> bm25_q) AS rank
            FROM filtered_chunks
            ORDER BY content <@> bm25_q
            LIMIT p_bm25_limit
        ),
        vector_results AS MATERIALIZED (
            SELECT id AS chunk_id,
                   ROW_NUMBER() OVER (ORDER BY embedding <=> p_query_embedding) AS rank
            FROM filtered_chunks
            ORDER BY embedding <=> p_query_embedding
            LIMIT p_vector_limit
        )
        SELECT
            c.id,
            c.document_id,
            c.content,
            c.section_path,
            c.chunk_level,
            c.metadata,
            b.rank::INT,
            v.rank::INT,
            ((p_bm25_weight * COALESCE(1.0 / (p_rrf_k + b.rank), 0)) +
             (p_vector_weight * COALESCE(1.0 / (p_rrf_k + v.rank), 0)))::FLOAT
        FROM filtered_chunks c
        LEFT JOIN bm25_results b ON c.id = b.chunk_id
        LEFT JOIN vector_results v ON c.id = v.chunk_id
        WHERE b.chunk_id IS NOT NULL OR v.chunk_id IS NOT NULL
        ORDER BY rrf_score DESC
        LIMIT p_match_count;
    END;
    $$ LANGUAGE plpgsql;
    """)

    # get_parent_context avec prise en compte de deleted_at
    await conn.execute("""
    CREATE OR REPLACE FUNCTION get_parent_context(p_chunk_id UUID, p_levels INT DEFAULT 1)
    RETURNS TABLE (chunk_id UUID, content TEXT, section_path TEXT[], chunk_level INT) AS $$
    WITH RECURSIVE parent_chain AS (
        SELECT id, parent_id, content, section_path, chunk_level, 0 AS depth
        FROM chunks
        WHERE id = p_chunk_id AND deleted_at IS NULL
        UNION ALL
        SELECT c.id, c.parent_id, c.content, c.section_path, c.chunk_level, pc.depth + 1
        FROM chunks c
        INNER JOIN parent_chain pc ON c.id = pc.parent_id
        WHERE pc.depth < p_levels AND c.deleted_at IS NULL
    )
    SELECT id, content, section_path, chunk_level
    FROM parent_chain
    WHERE id != p_chunk_id
    ORDER BY chunk_level ASC;
    $$ LANGUAGE SQL;
    """)
    log.info("sql_functions_created")