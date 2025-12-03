from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from backend.api.dependencies import get_rag_service_dependency
from backend.schemas.schemas import RAGRequest, RAGResponse, SearchRequest, SearchResponse
from backend.services.rag_service import RAGService

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/search", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def search_documents(
    request: SearchRequest,
    rag_service: RAGService = Depends(get_rag_service_dependency),
) -> SearchResponse:
    try:
        logger.info(f"Search: {request.query[:50]}")
        response = await rag_service.search(request)
        logger.info(f"Found {response.total_found} results")
        return response
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/ask", response_model=RAGResponse, status_code=status.HTTP_200_OK)
async def ask_question(
    request: RAGRequest,
    rag_service: RAGService = Depends(get_rag_service_dependency),
) -> RAGResponse:
    try:
        logger.info(f"Question: {request.question[:50]}")
        response = await rag_service.generate_answer(request)
        logger.info(f"Generated answer in {response.generation_time_ms:.0f}ms")
        return response
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
