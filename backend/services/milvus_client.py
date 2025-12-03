import threading
from typing import Any, List, Dict, Optional

from loguru import logger
from pymilvus import (
    connections, 
    Collection, 
    CollectionSchema, 
    FieldSchema, 
    DataType, 
    MilvusException, 
    utility
)

from backend.core.config import settings

class MilvusClient:
    def __init__(self) -> None:
        self.collection_name = settings.milvus.collection_name
        self.dimension = settings.milvus.dimension
        self.alias = "default"
        self._collection: Optional[Collection] = None
        self._lock = threading.RLock()
        self.output_fields = ["doc_id"] 

    def connect(self) -> None:
        with self._lock:
            try:
                if not connections.has_connection(self.alias):
                    connections.connect(
                        alias=self.alias, 
                        host=settings.milvus.host, 
                        port=settings.milvus.port
                    )
                    logger.info(f"Connected to Milvus at {settings.milvus.host}:{settings.milvus.port}")

                if utility.has_collection(self.collection_name):
                    self._collection = Collection(self.collection_name)
                    
                    # Check if collection has data before attempting to load
                    if self._collection.num_entities == 0:
                        logger.warning(f"Collection '{self.collection_name}' exists but is empty. Skipping load.")
                        return
                    
                    try:
                        self._collection.load()
                        logger.info(f"Collection '{self.collection_name}' loaded.")
                    except MilvusException as e:
                        # Index not found - try to build if we have enough data
                        if "index not found" in str(e) or e.code == 700:
                            logger.info("Index not found. Attempting to build...")
                            self._build_index_safe()
                            
                            # Only load if index was successfully built
                            if self._collection.has_index():
                                self._collection.load()
                                logger.info(f"Collection '{self.collection_name}' loaded after index build.")
                            else:
                                logger.warning(
                                    f"Collection '{self.collection_name}' has insufficient data "
                                    f"({self._collection.num_entities} < {settings.milvus.index_build_threshold}). "
                                    "Collection ready but not loaded."
                                )
                        else:
                            raise e
                else:
                    self._create_collection()
                    logger.info(f"Collection '{self.collection_name}' created. Waiting for data.")
                    
            except Exception as e:
                logger.error(f"Milvus init failed: {e}")
                raise

    def disconnect(self) -> None:
        with self._lock:
            if self._collection:
                self._collection.release()
                self._collection = None
            connections.disconnect(self.alias)

    def _create_collection(self) -> None:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="published_at", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields=fields, description="Persian News Vectors")
        self._collection = Collection(name=self.collection_name, schema=schema)
        logger.info(f"Collection '{self.collection_name}' created (Empty/No Index).")
        # Do NOT build index or load here. Wait for data.

    def _build_index_safe(self) -> None:
        """
        Safely builds index only if data exceeds threshold.
        """
        if not self._collection: return
        
        try:
            # Check approximate count
            if self._collection.num_entities < settings.milvus.index_build_threshold:
                logger.info(f"Skipping index build: Not enough entities ({self._collection.num_entities} < {settings.milvus.index_build_threshold})")
                return

            logger.info("Building Milvus Index...")
            index_params = {"nlist": settings.milvus.nlist}
            if settings.milvus.index_type == "IVF_PQ":
                index_params["m"] = settings.milvus.m 
                index_params["nbits"] = settings.milvus.nbits
            
            try:
                self._collection.create_index("embedding", {
                    "index_type": settings.milvus.index_type,
                    "metric_type": settings.milvus.metric_type,
                    "params": index_params,
                }, index_name="vector_index")
                logger.info("Vector index built successfully.")
            except Exception as e:
                logger.warning(f"Vector index creation warning: {e}")

            try:
                self._collection.create_index("doc_id", index_name="doc_id_index")
                logger.info("Scalar index built successfully.")
            except Exception as e:
                logger.warning(f"Scalar index creation warning: {e}")
                
        except Exception as e:
            logger.error(f"Index build failed: {e}")

    def insert_vectors(self, doc_ids: List[str], embeddings: List[List[float]], timestamps: List[int]) -> List[int]:
        if not self._collection: raise RuntimeError("Milvus disconnected")
        data = [doc_ids, embeddings, timestamps]
        with self._lock:
            result = self._collection.insert(data)
            
            # Auto-index if threshold reached and no index exists
            # We check if we passed the threshold with this insertion
            if self._collection.num_entities >= settings.milvus.index_build_threshold:
                if not self._collection.has_index():
                    self._build_index_safe()
                    # Now we can attempt to load for future searches
                    try:
                        self._collection.load()
                        logger.info("Collection loaded automatically after index build.")
                    except: pass

        return list(result.primary_keys)

    def search(self, query_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        if not self._collection: 
            raise RuntimeError("Milvus disconnected")
        
        with self._lock:
            # Early return if collection is empty
            if self._collection.num_entities == 0:
                logger.debug("Search skipped: Collection is empty.")
                return []
            
            try:
                # Only load if not already loaded
                if not hasattr(self._collection, '_is_loaded') or not self._collection._is_loaded:
                    self._collection.load()
                
                search_params = {
                    "metric_type": settings.milvus.metric_type, 
                    "params": {"nprobe": settings.milvus.nprobe}
                }
                
                results = self._collection.search(
                    data=[query_embedding],
                    anns_field="embedding",
                    param=search_params,
                    limit=top_k,
                    output_fields=self.output_fields,
                )
            except MilvusException as e:
                if "index not found" in str(e) or e.code == 700 or "not loaded" in str(e).lower():
                    logger.warning("Search failed: Collection not indexed/loaded yet.")
                    return []
                logger.error(f"Search error: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected search error: {e}")
                raise

        hits = []
        if results:
            for hit in results[0]:
                hits.append({
                    "doc_id": hit.entity.get("doc_id"),
                    "score": hit.score
                })
        return hits

    def get_vectors_by_doc_ids(self, doc_ids: List[str]) -> Dict[str, List[float]]:
        if not self._collection or not doc_ids:
            return {}

        ids_expr = str(doc_ids)
        try:
            with self._lock:
                # Same safety check as search
                try:
                    self._collection.load()
                except:
                    return {}
                    
                res = self._collection.query(
                    expr=f"doc_id in {ids_expr}",
                    output_fields=["doc_id", "embedding"]
                )
            return {item["doc_id"]: item["embedding"] for item in res}
        except Exception as e:
            logger.error(f"Failed to fetch vectors: {e}")
            return {}

    def health_check(self) -> Dict[str, Any]:
        if not self._collection: return {"status": "disconnected"}
        return {"status": "up", "approx_count": self._collection.num_entities}

_milvus_client: Optional[MilvusClient] = None
_init_lock = threading.Lock()

def get_milvus_client() -> MilvusClient:
    global _milvus_client
    if _milvus_client is None:
        with _init_lock:
            if _milvus_client is None:
                _milvus_client = MilvusClient()
    return _milvus_client
