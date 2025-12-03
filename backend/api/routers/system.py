from datetime import datetime
from fastapi import APIRouter, status
from backend.schemas.schemas import HealthCheck, SystemStats
from backend.services.milvus_client import get_milvus_client
from backend.services.elasticsearch_service import get_elasticsearch_service

router = APIRouter(prefix="/system", tags=["System"])

@router.get("/health", response_model=HealthCheck, status_code=status.HTTP_200_OK)
async def health_check() -> HealthCheck:
    
    # 1. Milvus
    milvus_client = get_milvus_client()
    m_health = milvus_client.health_check()
    m_ok = m_health.get("status") == "up"

    # 2. Elasticsearch
    es_service = get_elasticsearch_service()
    es_health = await es_service.health_check()
    es_status = es_health.get("status", "").lower()
    es_ok = es_status in ["green", "yellow"]

    # Overall
    overall = "healthy"
    if not m_ok or not es_ok:
        overall = "degraded"

    return HealthCheck(
        status=overall,
        timestamp=datetime.now(),
        components={
            "milvus": m_health,
            "elasticsearch": es_health
        }
    )

@router.get("/stats", response_model=SystemStats, status_code=status.HTTP_200_OK)
async def get_stats() -> SystemStats:
    es_service = get_elasticsearch_service()
    h = await es_service.health_check()
    
    return SystemStats(
        total_documents=h.get("total_documents", 0)
    )
