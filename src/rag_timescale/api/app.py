from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from structlog import get_logger

from rag_timescale.api import collections, documents, keys, permissions, search
from rag_timescale.config import settings

log = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from rag_timescale.db.connection import close_pool
    from rag_timescale.db.migrations import run_migrations
    from rag_timescale.embeddings.provider import get_embedding_dimensions

    await run_migrations()
    dimensions = get_embedding_dimensions()
    log.info("application_started", dimensions=dimensions)
    yield
    await close_pool()
    log.info("application_stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Timescale API",
        description="Enterprise RAG system with TimescaleDB, hybrid search, and intelligent chunking",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(keys.router)
    app.include_router(collections.router)
    app.include_router(documents.router)
    app.include_router(search.router)
    app.include_router(permissions.router)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app
