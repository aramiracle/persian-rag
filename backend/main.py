from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.api.routers import rag, system, upload, documents
from backend.core.config import settings
from backend.services.embedding_service import get_embedding_service
from backend.services.llm_service import get_llm_service
from backend.services.milvus_client import get_milvus_client
from backend.services.elasticsearch_service import get_elasticsearch_service

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info(f"Starting {settings.app_name} v{settings.version}")

    try:
        # 1. Connect Milvus
        milvus_client = get_milvus_client()
        milvus_client.connect()

        # 2. Connect & Init Elasticsearch
        es_service = get_elasticsearch_service()
        await es_service.initialize()

        # 3. Load Embedder
        embedding_service = get_embedding_service()
        embedding_service.initialize()

        # 4. LLM
        get_llm_service()

        logger.info("✅ All services initialized.")

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        raise

    yield

    logger.info("Shutting down")
    try:
        milvus_client.disconnect()
        embedding_service.cleanup()
        await es_service.close()
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag.router, prefix="/api/v1")
app.include_router(upload.router, prefix="/api/v1")
app.include_router(system.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Persian RAG API with Elasticsearch & Milvus"}
