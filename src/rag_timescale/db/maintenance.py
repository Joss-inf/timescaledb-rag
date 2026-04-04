import uuid
from rag_timescale.db.connection import get_pool

async def soft_delete_document(document_id: uuid.UUID):
    """Soft delete d'un document et de tous ses chunks."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("UPDATE documents SET deleted_at = NOW() WHERE id = $1", document_id)
            await conn.execute("UPDATE chunks SET deleted_at = NOW() WHERE document_id = $1", document_id)

async def purge_soft_deleted(delay_days: int = 30):
    """Suppression physique des lignes soft-deleted depuis plus de delay_days."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Supprimer par petits lots pour éviter des transactions trop grosses
        await conn.execute("""
            WITH deleted AS (
                DELETE FROM chunks
                WHERE deleted_at < NOW() - ($1 * INTERVAL '1 day')
                LIMIT 10000
                RETURNING id
            ) SELECT count(*) FROM deleted
        """, delay_days)
        # De même pour documents (après les chunks)
        await conn.execute("""
            DELETE FROM documents
            WHERE deleted_at < NOW() - ($1 * INTERVAL '1 day')
        """, delay_days)

async def vacuum_chunks():
    """Vacuum et analyse de la table chunks."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("VACUUM (ANALYZE) chunks")

async def optimize_bm25():
    """Force la fusion des segments BM25."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("SELECT bm25_force_merge('idx_chunks_bm25')")