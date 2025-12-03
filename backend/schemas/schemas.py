from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, HttpUrl, validator

# --- Core Data Models ---

class NewsDocument(BaseModel):
    """
    Represents a news article.
    Used for both Ingestion (Upload) and Retrieval (Response).
    """
    doc_id: Optional[str] = Field(default=None, description="Unique identifier (hash)")
    news_agency_name: Optional[str] = None
    url: Optional[str] = None  # Using str instead of HttpUrl to prevent validation errors on retrieval of legacy/malformed URLs
    title: Optional[str] = None
    content: Optional[str] = None
    published_at: Optional[datetime] = None
    
    # Metadata
    ner: Optional[Dict[str, List[str]]] = None
    topic: Optional[str] = None
    categories: List[str] = Field(default_factory=list)

    class Config:
        from_attributes = True


class DateRange(BaseModel):
    start: datetime
    end: datetime


# --- Search & Retrieval Schemas ---

class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Optional filters
    date_range: Optional[DateRange] = None
    agency_filter: Optional[str] = None


class SearchResult(BaseModel):
    """A ranked result returned from the search engine."""
    document: NewsDocument
    score: float
    rank: int


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_found: int
    retrieval_time_ms: float


# --- RAG (Generation) Schemas ---

class RAGRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)


class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: List[SearchResult]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float


# --- System & Management Schemas ---

class UploadResponse(BaseModel):
    message: str
    total_documents: int
    successful: int
    failed: int
    processing_time_ms: float


class ComponentHealth(BaseModel):
    status: str
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    components: Dict[str, Any]


class SystemStats(BaseModel):
    total_documents: int
