import uuid
import os
import time
import math
import csv
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Dict, Any, Optional

from loguru import logger
from backend.core.config import settings
from backend.services.milvus_client import MilvusClient
from backend.services.elasticsearch_service import ElasticsearchService
from backend.services.embedding_service import EmbeddingService

class JobStore:
    def __init__(self) -> None:
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def create_sync(self, job_id: str) -> None:
        self.jobs[job_id] = {
            "status": "processing", 
            "progress": 0, 
            "message": "Initializing...", 
            "start_time": time.time()
        }

    def update(self, job_id: str, progress: int, message: str, result: Optional[Dict[str, Any]] = None) -> None:
        if job_id in self.jobs:
            self.jobs[job_id]["progress"] = progress
            self.jobs[job_id]["message"] = message
            if result:
                self.jobs[job_id]["result"] = result
                self.jobs[job_id]["status"] = "completed"

    def fail(self, job_id: str, error: str) -> None:
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["message"] = error

    async def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

UPLOAD_JOBS = JobStore()

async def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    return await UPLOAD_JOBS.get(job_id)

class UploadService:
    def __init__(
        self, 
        milvus_client: MilvusClient,
        es_service: ElasticsearchService,
        embedding_service: EmbeddingService
    ) -> None:
        self.milvus = milvus_client
        self.es = es_service
        self.embedder = embedding_service
        # Thread pool for blocking operations
        self._executor = ThreadPoolExecutor(max_workers=3)

    def _count_total_rows_exact(self, file_path: str) -> int:
        """
        Counts logical CSV rows using the CSV parser.
        This correctly handles newlines inside quoted fields (multiline text).
        """
        try:
            # Using csv.reader is memory safe (streaming) and respects quotes.
            # We open with errors='ignore' to avoid crashing on encoding issues during simple counting.
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                # Efficiently count items in iterator
                row_count = sum(1 for _ in reader)
            
            # Subtract 1 for header
            return max(row_count - 1, 0)
        except Exception as e:
            logger.error(f"Error counting CSV rows: {e}")
            return 0

    async def process_csv_background(self, job_id: str, file_path: str) -> None:
        logger.info(f"Job {job_id}: Processing {file_path}")
        start_time = datetime.now()
        
        chunk_size = settings.upload.pandas_chunk_size
        total_processed = 0
        success_count = 0
        
        try:
            # --- STEP 1: Exact Row Count (Parser Aware) ---
            UPLOAD_JOBS.update(job_id, 0, "Scanning file for exact row count...")
            
            loop = asyncio.get_running_loop()
            total_rows = await loop.run_in_executor(
                self._executor, 
                self._count_total_rows_exact, 
                file_path
            )
            
            if total_rows == 0:
                total_chunks = 1
                logger.warning(f"Job {job_id}: File appears empty or header-only.")
            else:
                total_chunks = math.ceil(total_rows / chunk_size)

            logger.info(f"Job {job_id}: Logical Rows: {total_rows} | Total Chunks: {total_chunks}")
            
            # --- STEP 2: Process Chunks ---
            # We reopen the file with Pandas for processing
            with pd.read_csv(
                file_path, 
                chunksize=chunk_size, 
                on_bad_lines='skip', 
                dtype=str, 
                keep_default_na=False
            ) as reader:
                
                for chunk_num, df in enumerate(reader, start=1):
                    # Yield to event loop
                    await asyncio.sleep(0)

                    if df.empty:
                        continue
                    
                    # Calculate granular progress
                    # We define progress as "Starting to process chunk N"
                    current_percent = int(((chunk_num - 1) / total_chunks) * 100)
                    
                    UPLOAD_JOBS.update(
                        job_id, 
                        progress=current_percent, 
                        message=f"Preparing chunk {chunk_num}/{total_chunks} ({current_percent}%)"
                    )

                    # Process the chunk
                    chunk_success = await self._process_single_chunk(
                        df, 
                        job_id=job_id, 
                        chunk_num=chunk_num, 
                        total_chunks=total_chunks
                    )
                    
                    success_count += chunk_success
                    total_processed += len(df)
                    
            # --- STEP 3: Completion ---
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            UPLOAD_JOBS.update(
                job_id, 
                100, 
                "Done", 
                result={
                    "total_processed": total_processed, 
                    "success": success_count, 
                    "failed": total_processed - success_count, 
                    "time_ms": duration_ms
                }
            )
            logger.info(f"Job {job_id} finished. Total rows: {total_processed}")

        except Exception as e:
            logger.error(f"Job {job_id} crashed: {e}")
            UPLOAD_JOBS.fail(job_id, str(e))
        finally:
            if os.path.exists(file_path):
                try: os.remove(file_path)
                except: pass

    async def _process_single_chunk(
        self, 
        df: pd.DataFrame, 
        job_id: str, 
        chunk_num: int, 
        total_chunks: int
    ) -> int:
        """
        Processes a single dataframe chunk with granular status updates.
        """
        try:
            # Recalculate percent for sub-steps
            percent = int((chunk_num / total_chunks) * 100)
            percent = min(percent, 99)

            es_docs: List[Dict[str, Any]] = []
            milvus_data_buffer: Dict[str, List[Any]] = {"ids": [], "embeddings": [], "timestamps": []}
            texts_to_embed: List[str] = []
            
            df.columns = [c.lower().strip() for c in df.columns]

            # 1. Parsing
            for _, row in df.iterrows():
                title = str(row.get("title", "")).strip()
                url = str(row.get("url", "")).strip()
                content = str(row.get("content", "")).strip()
                agency = str(row.get("news_agency_name", "")).strip()
                
                if not content or not title: continue

                doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{url}-{title}"))
                
                try:
                    raw_date = row.get("published_at")
                    dt = pd.to_datetime(raw_date) if raw_date else datetime.now()
                    ts = int(dt.timestamp())
                    iso_date = dt.isoformat()
                except Exception:
                    dt = datetime.now()
                    ts = int(dt.timestamp())
                    iso_date = dt.isoformat()

                es_docs.append({
                    "doc_id": doc_id, "title": title, "content": content,
                    "url": url, "news_agency_name": agency,
                    "topic": str(row.get("topic", "")),
                    "categories": str(row.get("categories", "")),
                    "published_at": iso_date
                })
                
                texts_to_embed.append(content)
                milvus_data_buffer["ids"].append(doc_id)
                milvus_data_buffer["timestamps"].append(ts)

            if not es_docs: return 0

            # 2. Embedding
            UPLOAD_JOBS.update(
                job_id, 
                progress=percent, 
                message=f"Embedding chunk {chunk_num}/{total_chunks} ({percent}%)"
            )
            
            loop = asyncio.get_running_loop()
            vectors: List[List[float]] = await loop.run_in_executor(
                self._executor, 
                self.embedder.embed_documents, 
                texts_to_embed
            )
            
            if len(vectors) != len(es_docs): return 0
            milvus_data_buffer["embeddings"] = vectors

            # 3. Indexing
            UPLOAD_JOBS.update(
                job_id, 
                progress=percent, 
                message=f"Indexing chunk {chunk_num}/{total_chunks} ({percent}%)"
            )

            await loop.run_in_executor(
                self._executor,
                self.milvus.insert_vectors,
                milvus_data_buffer["ids"], 
                milvus_data_buffer["embeddings"], 
                milvus_data_buffer["timestamps"]
            )
            await self.es.bulk_index(es_docs)
            
            return len(es_docs)

        except Exception as e:
            logger.error(f"Chunk {chunk_num} error: {e}")
            return 0

def get_upload_service(
    milvus_client: MilvusClient, 
    es_service: ElasticsearchService, 
    embedding_service: EmbeddingService
) -> UploadService:
    return UploadService(milvus_client, es_service, embedding_service)
