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
        self._mgmt_lock = threading.RLock()
        self._index_built = False
        self.output_fields = ["doc_id"] 
        # Explicitly define the vector index name to prevent AmbiguousIndexName errors
        self.vector_index_name = "vector_index"

    def connect(self) -> None:
        with self._mgmt_lock:
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
                    
                    if self._collection.num_entities == 0:
                        logger.warning(f"Collection '{self.collection_name}' exists but is empty. Skipping load.")
                        return
                    
                    # FIX: Specify index_name to check specifically for the vector index
                    if self._collection.has_index(index_name=self.vector_index_name):
                        self._index_built = True
                    
                    try:
                        self._collection.load()
                        logger.info(f"Collection '{self.collection_name}' loaded. Entities: {self._collection.num_entities}")
                    except MilvusException as e:
                        # If index not found (or loaded), attempt to build/load safely
                        if "index not found" in str(e) or e.code == 700:
                            logger.info("Index not found. Attempting to build...")
                            self._build_index_safe()
                            if self._collection.has_index(index_name=self.vector_index_name):
                                self._collection.load()
                        else:
                            raise e
                else:
                    self._create_collection()
                    logger.info(f"Collection '{self.collection_name}' created. Waiting for data.")
                    
            except Exception as e:
                logger.error(f"Milvus init failed: {e}")
                raise

    def disconnect(self) -> None:
        with self._mgmt_lock:
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

    def _build_index_safe(self) -> None:
        """
        Thread-safe index building.
        """
        with self._mgmt_lock:
            if not self._collection:
                return
            
            if self._index_built:
                return
            
            try:
                # Check approximate count
                if self._collection.num_entities < settings.milvus.index_build_threshold:
                    return

                # FIX: Check specific index name
                if self._collection.has_index(index_name=self.vector_index_name):
                    self._index_built = True
                    return

                logger.info("Building Milvus Index...")
                index_params = {"nlist": settings.milvus.nlist}
                if settings.milvus.index_type == "IVF_PQ":
                    index_params["m"] = settings.milvus.m 
                    index_params["nbits"] = settings.milvus.nbits
                
                self._collection.create_index(
                    field_name="embedding",
                    index_params={
                        "index_type": settings.milvus.index_type,
                        "metric_type": settings.milvus.metric_type,
                        "params": index_params,
                    },
                    index_name=self.vector_index_name
                )
                self._index_built = True
                
                # Create scalar index for doc_id (optional, but good for hybrid logic)
                try:
                    self._collection.create_index(field_name="doc_id", index_name="doc_id_index")
                except Exception: 
                    pass
                
                logger.info("Milvus Index built successfully.")
                    
            except Exception as e:
                logger.error(f"Index build failed: {e}")

    def insert_vectors(self, doc_ids: List[str], embeddings: List[List[float]], timestamps: List[int]) -> List[int]:
        if not self._collection:
            raise RuntimeError("Milvus disconnected")
        
        try:
            data = [doc_ids, embeddings, timestamps]
            
            # 1. Insert (In-Memory)
            res = self._collection.insert(data)
            
            # 2. FORCE FLUSH (Kept per your request for live data visibility)
            # This ensures data is immediately visible in search at the cost of performance.
            self._collection.flush()
            
            logger.info(f"Milvus: Inserted {len(doc_ids)} vectors and Flushed.")

            # 3. Check Index
            if not self._index_built:
                if self._collection.num_entities >= settings.milvus.index_build_threshold:
                    self._build_index_safe()
                    with self._mgmt_lock:
                        if self._index_built:
                            try:
                                self._collection.load()
                            except: pass

            return list(res.primary_keys)
        except Exception as e:
            logger.error(f"Milvus insert failed: {e}")
            raise e

    def search(self, query_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        if not self._collection: 
            raise RuntimeError("Milvus disconnected")
        
        if self._collection.num_entities == 0:
            return []
        
        try:
            search_params = {
                "metric_type": settings.milvus.metric_type, 
                "params": {"nprobe": settings.milvus.nprobe}
            }
            
            # FIX: Specify index_name to ensure we use the Vector Index
            results = self._collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=self.output_fields,
                index_name=self.vector_index_name
            )
        except MilvusException as e:
            # Reload if needed (e.g., if index was dropped or segment released)
            if "not loaded" in str(e).lower() or e.code == 700:
                with self._mgmt_lock:
                    try:
                        self._collection.load()
                    except Exception as load_err:
                        logger.error(f"Failed to reload collection: {load_err}")
                        return []
                return self.search(query_embedding, top_k)
            logger.error(f"Search error: {e}")
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
            res = self._collection.query(
                expr=f"doc_id in {ids_expr}",
                output_fields=["doc_id", "embedding"]
            )
            return {item["doc_id"]: item["embedding"] for item in res}
        except Exception as e:
            logger.error(f"Failed to fetch vectors: {e}")
            return {}

    def health_check(self) -> Dict[str, Any]:
        if not self._collection:
            return {"status": "disconnected"}
        
        # This count is accurate because flush() is called on insert
        count = self._collection.num_entities
        return {"status": "up", "approx_count": count}

_milvus_client: Optional[MilvusClient] = None
_init_lock = threading.Lock()

def get_milvus_client() -> MilvusClient:
    global _milvus_client
    if _milvus_client is None:
        with _init_lock:
            if _milvus_client is None:
                _milvus_client = MilvusClient()
    return _milvus_client
