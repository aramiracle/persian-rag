from typing import List, Dict, Any, Optional, Tuple
from elasticsearch import AsyncElasticsearch, helpers
from loguru import logger
from backend.core.config import settings

class ElasticsearchService:
    def __init__(self) -> None:
        self.client = AsyncElasticsearch(
            hosts=settings.es.host,
            basic_auth=(settings.es.username, settings.es.password) if settings.es.username else None,
            request_timeout=settings.es.timeout,
            max_retries=3,
            retry_on_timeout=True
        )
        self.index_name = settings.es.index_name
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized: return

        try:
            exists = await self.client.indices.exists(index=self.index_name)
            
            # Persian-optimized configuration
            index_config = {
                "settings": {
                    "number_of_shards": 1, 
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "persian_optimized": {
                                "type": "persian" 
                            }
                        }
                    },
                    "similarity": { 
                        "default_bm25": { 
                            "type": "BM25", 
                            "b": settings.es.b,
                            "k1": settings.es.k1
                        } 
                    }
                },
                "mappings": {
                    "properties": {
                        "doc_id": {"type": "keyword"},
                        "content": { 
                            "type": "text", 
                            "analyzer": "persian_optimized", 
                            "similarity": "default_bm25" 
                        },
                        "title": { "type": "text", "analyzer": "persian_optimized" },
                        "url": {"type": "keyword"},
                        "news_agency_name": {"type": "keyword"},
                        "published_at": {"type": "date"},
                        "topic": {"type": "keyword"},
                        "categories": {"type": "keyword"}
                    }
                }
            }

            if exists:
                logger.info(f"Index '{self.index_name}' exists.")
            else:
                logger.info(f"Creating Elasticsearch index: {self.index_name} (Persian Analyzer)")
                await self.client.indices.create(index=self.index_name, body=index_config)
                
            self._initialized = True
        except Exception as e:
            logger.error(f"ES Initialization failed: {e}")
            raise

    async def close(self) -> None:
        await self.client.close()

    async def bulk_index(self, documents: List[Dict[str, Any]]) -> Tuple[int, int]:
        actions = [{ "_index": self.index_name, "_id": doc["doc_id"], "_source": doc } for doc in documents]
        try:
            success, failed = await helpers.async_bulk(self.client, actions, refresh=True)
            return success, len(failed) if isinstance(failed, list) else 0
        except Exception as e:
            logger.error(f"ES bulk index error: {e}")
            raise

    async def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not query or not query.strip(): return []
        
        # Uses multi_match with phrase_prefix for better Persian performance
        search_body = {
            "size": top_k,
            "_source": False,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "content"],
                    "type": "phrase_prefix"
                }
            }
        }
        try:
            resp = await self.client.search(
                index=self.index_name, 
                body=search_body
            )
            hits = resp['hits']['hits']
            return [(hit['_id'], hit['_score']) for hit in hits]
        except Exception as e:
            logger.error(f"ES Search failed: {e}")
            return []

    async def get_bm25_scores(self, query: str, doc_ids: List[str]) -> Dict[str, float]:
        if not query or not doc_ids: return {}
        
        search_body = {
            "size": len(doc_ids),
            "_source": False,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title", "content"],
                                "type": "phrase_prefix"
                            }
                        }
                    ],
                    "filter": [
                        {"ids": {"values": doc_ids}}
                    ]
                }
            }
        }
        
        try:
            resp = await self.client.search(
                index=self.index_name, 
                body=search_body
            )
            return {hit['_id']: hit['_score'] for hit in resp['hits']['hits']}
        except Exception as e:
            logger.error(f"ES Cross-Score failed: {e}")
            return {}

    async def get_documents_by_ids(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        if not doc_ids: return []
        try:
            resp = await self.client.mget(index=self.index_name, ids=doc_ids)
            results = []
            for doc in resp['docs']:
                if doc.get('found'):
                    data = doc['_source']
                    data['id'] = doc['_id']
                    results.append(data)
            return results
        except Exception as e:
            logger.error(f"ES Mget failed: {e}")
            return []
            
    async def health_check(self) -> Dict[str, Any]:
        try:
            health = await self.client.cluster.health()
            count = await self.client.count(index=self.index_name)
            return {
                "status": health.get("status", "unknown"),
                "total_documents": count.get("count", 0)
            }
        except Exception as e:
            return {"status": "down", "error": str(e)}

_es_service: Optional[ElasticsearchService] = None

def get_elasticsearch_service() -> ElasticsearchService:
    global _es_service
    if _es_service is None:
        _es_service = ElasticsearchService()
    return _es_service
