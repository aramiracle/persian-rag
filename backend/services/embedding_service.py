import os
import time
import threading
from typing import List, Optional
from collections import deque
import asyncio

import numpy as np
from loguru import logger

from backend.core.config import settings

LLAMA_POOLING_TYPE_MEAN = 1

class EmbeddingRequest:
    """
    Lightweight holder for request data and synchronization primitives.
    """
    __slots__ = ('texts', 'event', 'result', 'error', 'created_at')
    
    def __init__(self, texts: List[str]):
        self.texts = texts
        self.event = threading.Event()
        self.result = None
        self.error = None
        self.created_at = time.time()

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
        
        # --- Custom Priority Queue using Deque ---
        # High Priority: Search Queries (LIFO/FIFO doesn't matter much for small volume, FIFO used)
        self._high_prio_queue: deque = deque()
        # Low Priority: Document Upload Chunks (FIFO)
        self._low_prio_queue: deque = deque()
        
        # Synchronization
        self._cv = threading.Condition()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = False
        
        self._init_lock = threading.Lock()

    def initialize(self) -> None:
        with self._init_lock:
            if self._initialized:
                return
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found: {self.model_path}")

            from llama_cpp import Llama
            logger.info(f"Loading embedding model: {self.model_path} (Device: {self.device})")

            self._model = Llama(
                model_path=self.model_path,
                embedding=True,
                pooling_type=LLAMA_POOLING_TYPE_MEAN,
                n_gpu_layers=-1 if self.device == "cuda" else 0,
                n_ctx=8192,
                n_batch=self.batch_size,
                verbose=False,
            )
            self._initialized = True
            logger.info(f"Embedding model ready. Batch Size: {self.batch_size}")
            
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

    def _worker_loop(self) -> None:
        """
        Worker loop using Condition Variables for low-latency wakeups.
        Strictly prioritizes High Priority queue.
        """
        logger.info("Embedding worker loop started (Priority Deque Mode)")
        
        while True:
            req: Optional[EmbeddingRequest] = None
            
            with self._cv:
                # 1. Wait until there is work or stop signal
                while (not self._high_prio_queue and not self._low_prio_queue) and not self._stop_event:
                    self._cv.wait()
                
                if self._stop_event:
                    break

                # 2. Strict Priority Selection
                # Always check high priority first.
                if self._high_prio_queue:
                    req = self._high_prio_queue.popleft()
                elif self._low_prio_queue:
                    req = self._low_prio_queue.popleft()
            
            # 3. Process (Outside Lock to allow new submissions)
            if req:
                self._process_request_safe(req)

    def _process_request_safe(self, req: EmbeddingRequest) -> None:
        try:
            # Drop stale requests (> 5 mins old) to prevent queue clogging
            if time.time() - req.created_at > 300:
                logger.warning("Dropping stale embedding request")
                req.error = TimeoutError("Request dropped due to ttl")
                return

            # Compute logic
            embeddings = self._compute_batch(req.texts)
            req.result = embeddings
        except Exception as e:
            logger.error(f"Embedding processing failed: {e}")
            req.error = e
        finally:
            # Wake up the waiter
            req.event.set()

    def _compute_batch(self, texts: List[str]) -> List[List[float]]:
        if not self._initialized or not self._model:
            raise RuntimeError("Model not initialized")
        
        embeddings = []
        # Process in model's native batch size
        # This loop is usually short because 'texts' comes pre-chunked (micro-batches)
        # from embed_documents.
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # llama.cpp is not thread-safe, but only this worker calls it
            result = self._model.create_embedding(batch)
            
            vectors = np.array([item["embedding"] for item in result["data"]], dtype=np.float32)

            if self.dimension < vectors.shape[1]:
                vectors = vectors[:, :self.dimension]
            
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            
            embeddings.extend((vectors / norms).tolist())

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        High Priority: Adds to _high_prio_queue
        """
        if not text: return [0.0] * self.dimension
        
        processed = f"{self.query_prefix}{' '.join(text.split())}"
        req = EmbeddingRequest([processed])
        
        with self._cv:
            self._high_prio_queue.append(req)
            self._cv.notify() # Wake up worker immediately
        
        # Fast timeout for queries
        if req.event.wait(timeout=20.0):
            if req.error: raise req.error
            return req.result[0]
        else:
            raise TimeoutError("Embedding query timed out (System busy)")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Low Priority: Slices into micro-batches and adds to _low_prio_queue.
        Allows High Priority queries to jump the queue between batches.
        """
        if not texts: return []
        
        processed_texts = [f"{self.doc_prefix}{' '.join(t.split())}" for t in texts]
        
        # 1. Create requests for micro-batches
        # We break the large document list into small chunks (size of model batch)
        # This ensures the worker returns to the loop frequently to check for high-priority queries.
        micro_batches = []
        for i in range(0, len(processed_texts), self.batch_size):
            batch = processed_texts[i : i + self.batch_size]
            micro_batches.append(EmbeddingRequest(batch))
        
        # 2. Enqueue all batches
        with self._cv:
            for req in micro_batches:
                self._low_prio_queue.append(req)
            self._cv.notify(n=len(micro_batches))
            
        # 3. Wait for results sequentially
        # This blocks the calling thread (UploadService), but NOT the Embedding Worker.
        results = []
        try:
            for req in micro_batches:
                # Long timeout for bulk processing
                if req.event.wait(timeout=600.0):
                    if req.error: raise req.error
                    results.extend(req.result)
                else:
                    raise TimeoutError("Bulk embedding timed out")
            return results
        except Exception as e:
            logger.error(f"Bulk embed error: {e}")
            raise e

    def cleanup(self) -> None:
        with self._init_lock:
            with self._cv:
                self._stop_event = True
                self._cv.notify_all()
            
            if self._worker_thread:
                self._worker_thread.join(timeout=2.0)
            
            if self._model:
                del self._model
                self._model = None
            self._initialized = False

    def health_check(self) -> dict:
        return {
            "status": "up" if self._initialized else "down",
            "high_prio_pending": len(self._high_prio_queue),
            "low_prio_pending": len(self._low_prio_queue)
        }

_service: Optional[EmbeddingService] = None
_global_init_lock = threading.Lock()

def get_embedding_service() -> EmbeddingService:
    global _service
    if _service is None:
        with _global_init_lock:
            if _service is None:
                _service = EmbeddingService()
    return _service
