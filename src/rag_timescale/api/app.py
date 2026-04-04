from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from structlog import get_logger

from rag_timescale.api import collections, documents, keys, permissions, search
from rag_timescale.config import settings

log = get_logger()

# --- GESTION DU CYCLE DE VIE ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le démarrage et l'arrêt de l'application :
    - Initialisation du pool de connexion
    - Exécution des migrations SQL
    - Vérification du fournisseur d'embeddings
    """
    from rag_timescale.db.connection import close_pool, get_pool
    from rag_timescale.db.migrations import run_migrations
    from rag_timescale.embeddings.provider import get_embedding_dimensions

    try:
        # 1. Connexion DB
        await get_pool()
        
        # 2. Setup DB (Tables, Index, Hyper-tables Timescale)
        await run_migrations()
        
        # 3. Validation Provider Embeddings (OpenAI, Mistral, Local...)
        dimensions = get_embedding_dimensions()
        log.info("application_started", dimensions=dimensions, status="ready")
    except Exception as e:
        log.error("application_startup_failed", error=str(e))
        # On lève l'exception pour empêcher l'API de tourner dans un état invalide
        raise e

    yield
    
    # 4. Nettoyage
    await close_pool()
    log.info("application_stopped")


# --- FACTORY DE L'APPLICATION ---
def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Timescale API",
        description="Enterprise RAG system with TimescaleDB, hybrid search, and intelligent chunking",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    # --- SÉCURITÉ : CORS ---
    cors_config = {
        "allow_origins": settings.api.cors_origins,
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }
    
    # Alerte de sécurité si la config est trop permissive en prod
    if settings.api.cors_origins == ["*"] and cors_config["allow_credentials"]:
        log.warning("cors_security_risk", detail="Allowing all origins with credentials is risky")

    app.add_middleware(CORSMiddleware, **cors_config)

    # --- SÉCURITÉ : GESTION GLOBALE DES ERREURS ---
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        log.error("unhandled_exception", error=str(exc), path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error. Our team has been notified.",
                "type": "server_error"
            }
        )

    # --- ROUTAGE (Centralisé) ---
    V1_PREFIX = "/api/v1"

    # Authentification & Gestion des clés
    app.include_router(keys.router, prefix=f"{V1_PREFIX}/keys", tags=["Authentication"])

    # Collections (Ressource principale)
    app.include_router(collections.router, prefix=f"{V1_PREFIX}/collections", tags=["Collections"])

    # Recherche (Branchée par collection pour l'isolation)
    app.include_router(search.router, prefix=f"{V1_PREFIX}/collections/{{collection_id}}/search", tags=["Search"])

    # Documents & Ingestion (Sous-ressource de collection)
    app.include_router(documents.router, prefix=f"{V1_PREFIX}/collections/{{collection_id}}/documents", tags=["Documents"])

    # Permissions & Partage
    app.include_router(permissions.router, prefix=f"{V1_PREFIX}/collections/{{collection_id}}/permissions", tags=["Permissions"])

    # --- ENDPOINTS SYSTÈME ---
    @app.get("/health", tags=["System"])
    async def health_check():
        """Vérifie l'état de l'API et de la base de données."""
        from rag_timescale.db.connection import get_pool
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute("SELECT 1")
            return {
                "status": "healthy",
                "components": {
                    "database": "connected",
                    "api": "up"
                }
            }
        except Exception as e:
            log.error("health_check_failed", error=str(e))
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "components": {"database": "disconnected"}
                }
            )

    return app

# Initialisation pour les serveurs ASGI (Uvicorn / Gunicorn)
app = create_app()