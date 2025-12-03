import os
import threading
from typing import List, Optional

import numpy as np
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.config import settings

LLAMA_POOLING_TYPE_MEAN = 1


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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def _process_batch(self, texts: List[str]) -> List[List[float]]:
        if not self._initialized:
            raise RuntimeError("Model not initialized")
        if not texts:
            return []

        embeddings = []
        # Removed lock to allow parallel calls if supported by runtime/hardware
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            result = self._model.create_embedding(batch)
            vectors = np.array([item["embedding"] for item in result["data"]], dtype=np.float32)

            if self.dimension < vectors.shape[1]:
                vectors = vectors[:, :self.dimension]

            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings.extend((vectors / norms).tolist())

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        processed = [f"{self.doc_prefix}{' '.join(t.split())}" for t in texts]
        return self._process_batch(processed)

    def embed_query(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.dimension
        processed = f"{self.query_prefix}{' '.join(text.split())}"
        return self._process_batch([processed])[0]

    def cleanup(self) -> None:
        with self._init_lock:
            if self._model:
                del self._model
                self._model = None
            self._initialized = False

    def health_check(self) -> dict:
        if not self._initialized:
            return {"status": "down", "error": "Not initialized"}
        try:
            self.embed_query("test")
            return {"status": "up", "dimension": self.dimension}
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
