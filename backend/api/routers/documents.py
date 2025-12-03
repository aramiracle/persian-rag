from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from loguru import logger

from backend.services.elasticsearch_service import ElasticsearchService, get_elasticsearch_service

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.get("/preview", status_code=status.HTTP_200_OK)
async def get_document_preview(
    limit: int = Query(10, ge=1, le=100),
    es_service: ElasticsearchService = Depends(get_elasticsearch_service),
) -> list[dict[str, Any]]:
    """
    Retrieve recent documents from Elasticsearch.
    """
    try:
        # Match All query for preview
        body = {
            "size": limit,
            "query": {"match_all": {}},
            "_source": ["doc_id", "title", "news_agency_name", "published_at", "topic"]
        }
        resp = await es_service.client.search(index=es_service.index_name, body=body)
        
        docs = []
        for hit in resp['hits']['hits']:
            d = hit['_source']
            d['id'] = hit['_id']
            docs.append(d)
        return docs

    except Exception as e:
        logger.error(f"Failed to fetch preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))
