import os
from functools import lru_cache
from typing import Literal, List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MilvusSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MILVUS_", env_file=".env", extra="ignore")

    host: str = Field(default="milvus")
    port: int = Field(default=19530)
    collection_name: str = Field(default="persian_news")
    
    dimension: int = Field(default=1024)
    index_type: str = Field(default="IVF_PQ") 
    metric_type: str = Field(default="IP")
    nlist: int = Field(default=65536)
    nprobe: int = Field(default=32)
    m: int = Field(default=64)
    nbits: int = Field(default=8)
    
    # Avoid building index on empty/small data
    index_build_threshold: int = Field(default=1024)


class ElasticsearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ES_", env_file=".env", extra="ignore")

    host: str = Field(default="http://elasticsearch:9200")
    username: str | None = Field(default=None)
    password: str | None = Field(default=None)
    index_name: str = Field(default="persian_news_v1")
    
    b: float = Field(default=0.75)
    k1: float = Field(default=1.2)
    timeout: int = Field(default=60)


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", env_file=".env", extra="ignore")

    model_path: str = Field(default="models/qwen3-embed-0.6b-q4_k_m.gguf")
    dimension: int = Field(default=1024)
    batch_size: int = Field(default=64)
    device: Literal["cuda", "cpu"] = Field(default="cpu")


class UploadSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="UPLOAD_", env_file=".env", extra="ignore")

    pandas_chunk_size: int = Field(default=1000)
    temp_dir: str = Field(default="storage/temp_uploads")


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")

    api_key: str 
    base_url: str = Field(default="https://api.openai.com/v1")
    model: str = Field(default="gpt-5.1")
    temperature: float = Field(default=0.1)
    max_completion_tokens: int = Field(default=2048)
    
    # Default to 400k (GPT-5.1 spec) if not set
    context_window: int = Field(default=400000)


class RAGSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    top_k: int = Field(default=10)
    rrf_k: int = Field(default=60)
    alpha: float = Field(default=0.5)
    retrieval_multiplier: int = Field(default=5)


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    app_name: str = Field(default="Persian News RAG (Scale)")
    version: str = Field(default="2.0.0")
    debug: bool = Field(default=False)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    cors_origins: List[str] = Field(default=["*"])

    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    es: ElasticsearchSettings = Field(default_factory=ElasticsearchSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    upload: UploadSettings = Field(default_factory=UploadSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()

settings = get_settings()
