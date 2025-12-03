import os
import time
import math
import threading
import psutil
from typing import List, Optional
from multiprocessing import cpu_count, Queue, Process
import queue
import numpy as np
from loguru import logger

from backend.core.config import settings

LLAMA_POOLING_TYPE_MEAN = 1
BYTES_PER_FLOAT = 4 

class EmbeddingWorker(Process):
    def __init__(
        self,
        worker_id: int,
        task_queue: Queue,
        result_queue: Queue,
        model_path: str,
        device: str,
        batch_size: int,
        dimension: int,
        context_length: int
    ):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.dimension = dimension
        self.context_length = context_length
        
    def run(self):
        try:
            from llama_cpp import Llama
            
            logger.info(f"Worker {self.worker_id}: Loading model (mmap=True)...")
            
            self.model = Llama(
                model_path=self.model_path,
                embedding=True,
                pooling_type=LLAMA_POOLING_TYPE_MEAN,
                n_gpu_layers=-1 if self.device == "cuda" else 0,
                n_ctx=self.context_length,
                n_batch=self.batch_size,
                verbose=False,
                use_mmap=True,      
                use_mlock=False,    
                n_threads=1, 
            )
            
            logger.info(f"Worker {self.worker_id}: Ready")
            
            while True:
                try:
                    task = self.task_queue.get(timeout=1.0)
                    if task is None: 
                        logger.info(f"Worker {self.worker_id}: Shutting down")
                        break
                    
                    task_id, texts, prefix = task
                    processed = [f"{prefix}{' '.join(t.split())}" for t in texts]
                    
                    embeddings = []
                    for i in range(0, len(processed), self.batch_size):
                        batch = processed[i:i + self.batch_size]
                        result = self.model.create_embedding(batch)
                        
                        vectors = np.array(
                            [item["embedding"] for item in result["data"]], 
                            dtype=np.float32
                        )
                        
                        if self.dimension < vectors.shape[1]:
                            vectors = vectors[:, :self.dimension]
                        
                        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                        norms[norms == 0] = 1
                        embeddings.extend((vectors / norms).tolist())
                    
                    self.result_queue.put((task_id, embeddings, None))
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Worker {self.worker_id}: Error - {e}")
                    self.result_queue.put((task_id, None, str(e)))
                    
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Fatal Init Error - {e}")


class EmbeddingService:
    def __init__(self) -> None:
        # Load Directly from Config
        self.model_path = settings.embedding.model_path
        self.device = settings.embedding.device
        self.batch_size = settings.embedding.batch_size
        self.dimension = settings.embedding.dimension
        
        self.doc_prefix = settings.embedding.document_prefix
        self.query_prefix = settings.embedding.query_prefix
        
        # Resource Params
        self.context_length = settings.embedding.context_length
        self.model_layers = settings.embedding.model_layers
        self.ram_safety_margin = settings.embedding.ram_safety_margin
        self.worker_overhead_base = settings.embedding.worker_overhead_mb
        
        self._initialized = False
        self._init_lock = threading.Lock()
        
        self.num_workers = self._calculate_optimal_worker_count()
        
        self._task_queue: Optional[Queue] = None
        self._result_queue: Optional[Queue] = None
        self._workers: List[EmbeddingWorker] = []
        self._next_task_id = 0
        self._task_id_lock = threading.Lock()
        self._query_model = None

    def _calculate_optimal_worker_count(self) -> int:
        """
        Calculates maximum worker count based on RIGOROUS memory estimation.
        """
        # --- 1. CPU Constraint ---
        total_cores = cpu_count()
        
        # --- 2. Memory Constraint ---
        mem = psutil.virtual_memory()
        total_ram_mb = mem.total / (1024 ** 2)
        available_ram_mb = mem.available / (1024 ** 2)
        
        try:
            model_size_mb = os.path.getsize(self.model_path) / (1024 ** 2)
        except OSError:
            logger.warning("Could not determine model size. Assuming 4GB.")
            model_size_mb = 4096

        # --- MEMORY FORMULA ---
        
        # A. KV Cache Calculation (Assuming Standard Attention - Worst Case)
        # 2 tensors (K, V) * Layers * Dim * Context * 2 bytes (float16)
        kv_cache_mb = (self.model_layers * self.dimension * self.context_length * 4) / (1024 ** 2)
        
        # B. Compute Graph Safety Buffer
        # Large contexts require large temporary buffers for attention scores
        compute_buffer_mb = 512
        
        # C. Batch Processing Overhead
        batch_memory_mb = (self.batch_size * self.dimension * BYTES_PER_FLOAT * 4) / (1024 ** 2)
        
        # Total per Worker
        worker_total_mem_mb = (
            self.worker_overhead_base + 
            kv_cache_mb + 
            compute_buffer_mb + 
            batch_memory_mb
        )
        
        # System Reserve
        system_reserve_mb = total_ram_mb * self.ram_safety_margin
        
        # Usable RAM
        usable_ram_mb = available_ram_mb - system_reserve_mb - model_size_mb
        
        if usable_ram_mb <= 0:
            logger.critical("Not enough RAM to run even 1 worker comfortably!")
            return 1
            
        max_ram_workers = int(usable_ram_mb / worker_total_mem_mb)
        
        # --- Decision ---
        optimal_workers = min(total_cores, max_ram_workers)
        optimal_workers = max(1, optimal_workers)
        
        logger.info("--- STRICT Memory Resource Calculation ---")
        logger.info(f"System RAM:         {total_ram_mb:.0f} MB")
        logger.info(f"Available RAM:      {available_ram_mb:.0f} MB")
        logger.info(f"Model Size (mmap):  {model_size_mb:.0f} MB")
        logger.info(f"Reserve (OS+Model): {(system_reserve_mb + model_size_mb):.0f} MB")
        logger.info(f"Usable for Workers: {usable_ram_mb:.0f} MB")
        logger.info("-" * 40)
        logger.info(f"Context: {self.context_length} | Layers: {self.model_layers} | Dim: {self.dimension}")
        logger.info(f"1. Base Overhead:   {self.worker_overhead_base} MB")
        logger.info(f"2. KV Cache:        {kv_cache_mb:.0f} MB (Major Factor)")
        logger.info(f"3. Compute Buffer:  {compute_buffer_mb} MB")
        logger.info(f"TOTAL Per Worker:   ~{worker_total_mem_mb:.0f} MB")
        logger.info("-" * 40)
        logger.info(f"Max RAM Workers:    {max_ram_workers}")
        logger.info(f"CPU Cores:          {total_cores}")
        logger.info(f"FINAL COUNT:        {optimal_workers}")
        logger.info("------------------------------------------")
        
        return optimal_workers

    def initialize(self) -> None:
        with self._init_lock:
            if self._initialized: return
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found: {self.model_path}")

            logger.info(f"Initializing {self.num_workers} embedding workers...")
            
            self._task_queue = Queue(maxsize=self.num_workers * 2)
            self._result_queue = Queue()
            
            for i in range(self.num_workers):
                w = EmbeddingWorker(
                    worker_id=i,
                    task_queue=self._task_queue,
                    result_queue=self._result_queue,
                    model_path=self.model_path,
                    device=self.device,
                    batch_size=self.batch_size,
                    dimension=self.dimension,
                    context_length=self.context_length
                )
                w.start()
                self._workers.append(w)
            
            from llama_cpp import Llama
            self._query_model = Llama(
                model_path=self.model_path,
                embedding=True,
                pooling_type=LLAMA_POOLING_TYPE_MEAN,
                n_gpu_layers=0,
                n_ctx=self.context_length,
                n_batch=self.batch_size,
                verbose=False,
                use_mmap=True,
                use_mlock=False,
            )
            
            self._initialized = True
            logger.info("✅ Embedding Service Initialized")

    def embed_query(self, text: str) -> List[float]:
        if not self._initialized: raise RuntimeError("Not initialized")
        if not text: return [0.0] * self.dimension
        
        processed = f"{self.query_prefix}{' '.join(text.split())}"
        result = self._query_model.create_embedding([processed])
        vectors = np.array([item["embedding"] for item in result["data"]], dtype=np.float32)
        
        if self.dimension < vectors.shape[1]:
            vectors = vectors[:, :self.dimension]
        
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return (vectors / norms)[0].tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self._initialized: raise RuntimeError("Not initialized")
        if not texts: return []
        
        total_texts = len(texts)
        num_chunks = self.num_workers
        base_chunk_size = total_texts // num_chunks
        remainder = total_texts % num_chunks
        
        logger.info(f"Splitting {total_texts} docs -> {num_chunks} workers (Static)")
        
        start_time = time.time()
        task_ids = []
        
        current_idx = 0
        for i in range(num_chunks):
            size = base_chunk_size + (1 if i < remainder else 0)
            if size == 0: continue 
            
            chunk = texts[current_idx : current_idx + size]
            current_idx += size
            
            with self._task_id_lock:
                task_id = self._next_task_id
                self._next_task_id += 1
            
            self._task_queue.put((task_id, chunk, self.doc_prefix))
            task_ids.append(task_id)
        
        results = {}
        errors = []
        
        for _ in range(len(task_ids)):
            try:
                # Timeout safety (allowing for long context processing)
                timeout_val = 120 + (total_texts * 3) 
                task_id, embeddings, error = self._result_queue.get(timeout=timeout_val)
                
                if error:
                    errors.append(error)
                else:
                    results[task_id] = embeddings
            except queue.Empty:
                raise TimeoutError("Embedding Worker Timeout")

        if errors:
            raise RuntimeError(f"Embedding failed: {errors[0]}")
        
        final_embeddings = []
        for task_id in task_ids:
            final_embeddings.extend(results[task_id])
            
        elapsed = time.time() - start_time
        fps = total_texts / elapsed if elapsed > 0 else 0
        logger.info(f"✅ Finished {total_texts} docs in {elapsed:.2f}s ({fps:.1f} docs/s)")
        
        return final_embeddings

    def cleanup(self):
        with self._init_lock:
            if not self._initialized: return
            logger.info("Stopping workers...")
            for _ in self._workers: self._task_queue.put(None)
            for w in self._workers: w.join()
            self._workers = []
            self._query_model = None
            self._initialized = False

    def health_check(self) -> dict:
        return {
            "status": "up" if self._initialized else "down",
            "workers": self.num_workers,
            "device": self.device,
            "context_length": self.context_length
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
