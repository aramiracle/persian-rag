import os
import time
import threading
from typing import List, Optional
from queue import Queue, Empty
from dataclasses import dataclass

import numpy as np
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.config import settings

LLAMA_POOLING_TYPE_MEAN = 1


@dataclass
class EmbeddingRequest:
    """Request wrapper for priority queue"""
    texts: List[str]
    is_query: bool  # True for search queries (high priority)
    result_queue: Queue
    request_id: str


class EmbeddingService:
    def __init__(self) -> None:
        self.model_path = settings.embedding.model_path
        self.device = settings.embedding.device
        self.batch_size = settings.embedding.batch_size
        self.dimension = settings.embedding.dimension
        self.doc_prefix = getattr(settings.embedding, 'document_prefix', "passage: ")
        self.query_prefix = getattr(settings.embedding, 'query_prefix', "query: ")
        self._model = None
        self._initialized = False
        
        # Separate queues for queries and documents
        self._query_queue: Queue = Queue()
        self._doc_queue: Queue = Queue()
        
        # Worker thread
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Lock only for initialization
        self._init_lock = threading.Lock()

    def initialize(self) -> None:
        with self._init_lock:
            if self._initialized:
                return
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found: {self.model_path}")

            from llama_cpp import Llama
            logger.info(f"Loading embedding model: {self.model_path}")

            self._model = Llama(
                model_path=self.model_path,
                embedding=True,
                pooling_type=LLAMA_POOLING_TYPE_MEAN,
                n_gpu_layers=-1 if self.device == "cuda" else 0,
                n_ctx=32768,
                n_batch=self.batch_size,
                verbose=False,
            )
            self._initialized = True
            logger.info(f"Embedding model ready (dim={self.dimension})")
            
            # Start worker thread
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()
            logger.info("Embedding worker thread started")

    def _worker_loop(self) -> None:
        """
        Worker thread that processes embedding requests.
        Prioritizes query requests over document batches.
        """
        logger.info("Embedding worker loop started")
        
        while not self._stop_event.is_set():
            try:
                # Check query queue first (high priority)
                try:
                    req = self._query_queue.get(timeout=0.01)
                    self._process_request(req)
                    continue
                except Empty:
                    pass
                
                # Then check document queue (lower priority)
                try:
                    req = self._doc_queue.get(timeout=0.1)
                    self._process_request(req)
                except Empty:
                    continue
                    
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(0.1)
        
        logger.info("Embedding worker loop stopped")

    def _process_request(self, req: EmbeddingRequest) -> None:
        """Process a single embedding request"""
        try:
            embeddings = self._compute_embeddings(req.texts)
            req.result_queue.put(("success", embeddings))
        except Exception as e:
            logger.error(f"Embedding computation failed: {e}")
            req.result_queue.put(("error", str(e)))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def _compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Core embedding computation without locks.
        Only called by worker thread, so no race conditions.
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized")
        if not texts:
            return []

        embeddings = []
        
        # Process in mini-batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # No lock needed - only worker thread calls this
            result = self._model.create_embedding(batch)
            vectors = np.array([item["embedding"] for item in result["data"]], dtype=np.float32)

            if self.dimension < vectors.shape[1]:
                vectors = vectors[:, :self.dimension]

            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings.extend((vectors / norms).tolist())

        return embeddings

    def _submit_request(self, texts: List[str], is_query: bool, timeout: float = 30.0) -> List[List[float]]:
        """Submit request to appropriate queue and wait for result"""
        if not self._initialized:
            raise RuntimeError("Model not initialized")
        
        result_queue = Queue()
        req = EmbeddingRequest(
            texts=texts,
            is_query=is_query,
            result_queue=result_queue,
            request_id=f"{time.time()}"
        )
        
        # Route to appropriate queue
        if is_query:
            self._query_queue.put(req)
        else:
            self._doc_queue.put(req)
        
        # Wait for result
        try:
            status, result = result_queue.get(timeout=timeout)
            if status == "error":
                raise RuntimeError(f"Embedding failed: {result}")
            return result
        except Empty:
            raise TimeoutError(f"Embedding request timed out after {timeout}s")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Used by Upload Service (Bulk) - Lower priority
        """
        if not texts:
            return []
        processed = [f"{self.doc_prefix}{' '.join(t.split())}" for t in texts]
        return self._submit_request(processed, is_query=False, timeout=60.0)

    def embed_query(self, text: str) -> List[float]:
        """
        Used by Search/RAG (Single) - High priority
        """
        if not text or not text.strip():
            return [0.0] * self.dimension
        processed = f"{self.query_prefix}{' '.join(text.split())}"
        
        # Queries get priority and fast processing
        embeddings = self._submit_request([processed], is_query=True, timeout=10.0)
        return embeddings[0] if embeddings else [0.0] * self.dimension

    def cleanup(self) -> None:
        with self._init_lock:
            # Stop worker thread
            if self._worker_thread and self._worker_thread.is_alive():
                self._stop_event.set()
                self._worker_thread.join(timeout=5.0)
            
            if self._model:
                del self._model
                self._model = None
            self._initialized = False

    def health_check(self) -> dict:
        if not self._initialized:
            return {"status": "down", "error": "Not initialized"}
        try:
            # Quick health check
            test_emb = self.embed_query("test")
            return {
                "status": "up", 
                "dimension": self.dimension,
                "worker_alive": self._worker_thread.is_alive() if self._worker_thread else False,
                "query_queue_size": self._query_queue.qsize(),
                "doc_queue_size": self._doc_queue.qsize()
            }
        except Exception as e:
            return {"status": "down", "error": str(e)}


_service: Optional[EmbeddingService] = None
_global_init_lock = threading.Lock()

def get_embedding_service() -> EmbeddingService:
    global _service
    if _service is None:
        with _global_init_lock:
            if _service is None:
                _service = EmbeddingService()
    return _service
