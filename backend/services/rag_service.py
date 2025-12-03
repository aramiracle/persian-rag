import time
import asyncio
import ast
import numpy as np
from typing import Dict

from loguru import logger

from backend.schemas.schemas import (
    RAGRequest, RAGResponse, SearchRequest, SearchResponse, SearchResult
)
from backend.services.embedding_service import EmbeddingService
from backend.services.llm_service import LLMService
from backend.services.milvus_client import MilvusClient
from backend.services.elasticsearch_service import ElasticsearchService
from backend.core.config import settings
from backend.core.ranking import Ranker, FusionConfig

class RAGService:
    def __init__(
        self,
        milvus_client: MilvusClient,
        es_service: ElasticsearchService,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
    ) -> None:
        self.milvus = milvus_client
        self.es = es_service
        self.embedder = embedding_service
        self.llm = llm_service
        self.ranker = Ranker()
        
        self.config = FusionConfig(
            rrf_k=settings.rag.rrf_k,
            alpha=settings.rag.alpha,
            retrieval_multiplier=settings.rag.retrieval_multiplier
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        t_start = time.time()
        
        # 1. Initial Retrieval (Fetch more candidates)
        fetch_k = request.top_k * self.config.retrieval_multiplier
        
        # Helper for Parallel Retrieval
        query_vec = []
        
        async def run_dense():
            nonlocal query_vec
            try:
                # Embed query here so we can reuse vector for cross-scoring later
                query_vec = self.embedder.embed_query(request.query)
                return self.milvus.search(query_embedding=query_vec, top_k=fetch_k)
            except Exception as e:
                logger.error(f"Dense search failed: {e}")
                return []

        async def run_sparse():
            try:
                return await self.es.search(query=request.query, top_k=fetch_k)
            except Exception as e:
                logger.error(f"Sparse search failed: {e}")
                return []

        results = await asyncio.gather(run_dense(), run_sparse())
        dense_hits = results[0] # List[Dict(doc_id, score)]
        sparse_hits = results[1] # List[Tuple(doc_id, score)]
        
        # 2. Identify Candidates & Missing Scores
        all_ids = set([h['doc_id'] for h in dense_hits] + [h[0] for h in sparse_hits])
        
        # Maps to store final complete scores
        final_dense_scores: Dict[str, float] = {h['doc_id']: h['score'] for h in dense_hits}
        final_sparse_scores: Dict[str, float] = {h[0]: h[1] for h in sparse_hits}
        
        # Identify missing IDs
        missing_in_dense = [did for did in all_ids if did not in final_dense_scores]
        missing_in_sparse = [did for did in all_ids if did not in final_sparse_scores]
        
        # 3. Cross-Scoring (Hydration)
        
        # Task A: Get missing vectors from Milvus & Calc Cosine
        async def fill_dense():
            if not missing_in_dense or not query_vec: return
            try:
                # Fetch vectors
                id_to_vec = self.milvus.get_vectors_by_doc_ids(missing_in_dense)
                
                # Manual Dot Product (Cosine Sim if normalized)
                q_arr = np.array(query_vec)
                for doc_id, emb in id_to_vec.items():
                    d_arr = np.array(emb)
                    # Dot product
                    score = float(np.dot(q_arr, d_arr))
                    final_dense_scores[doc_id] = score
            except Exception as e:
                logger.error(f"Dense filling failed: {e}")

        # Task B: Get missing BM25 scores from ES
        async def fill_sparse():
            if not missing_in_sparse: return
            try:
                # Ask ES to score these specific IDs against the query
                new_scores = await self.es.get_bm25_scores(request.query, missing_in_sparse)
                # Update map (if doc not found in ES at all, score remains 0 or ignored?)
                # Actually, if not found in ES, score is effectively 0
                for doc_id in missing_in_sparse:
                    final_sparse_scores[doc_id] = new_scores.get(doc_id, 0.0)
            except Exception as e:
                logger.error(f"Sparse filling failed: {e}")

        # Execute filling in parallel
        await asyncio.gather(fill_dense(), fill_sparse())
        
        # Ensure everyone has a score (fill defaults if DB fetch failed)
        for doc_id in all_ids:
            if doc_id not in final_dense_scores: final_dense_scores[doc_id] = 0.0
            if doc_id not in final_sparse_scores: final_sparse_scores[doc_id] = 0.0

        # 4. Fusion
        ranked_tuples = self.ranker.fuse_full_scores(
            final_dense_scores, 
            final_sparse_scores, 
            self.config
        )

        # 5. Filter & Slice
        if request.min_score > 0:
            ranked_tuples = [x for x in ranked_tuples if x[1] >= request.min_score]
        
        top_tuples = ranked_tuples[:request.top_k]
        
        # 6. Content Hydration
        top_ids = [doc_id for doc_id, _ in top_tuples]
        docs_data = await self.es.get_documents_by_ids(top_ids)
        docs_map = {d['id']: d for d in docs_data}

        final_results = []
        for rank, (doc_id, score) in enumerate(top_tuples, start=1):
            doc_content = docs_map.get(doc_id)
            if doc_content:
                # Parse categories
                cats = doc_content.get("categories")
                if isinstance(cats, str):
                    try:
                        if cats.strip().startswith("[") and cats.strip().endswith("]"):
                            doc_content["categories"] = ast.literal_eval(cats)
                        else:
                            doc_content["categories"] = [c.strip() for c in cats.split(",") if c.strip()]
                    except: doc_content["categories"] = []
                elif cats is None: doc_content["categories"] = []

                final_results.append(SearchResult(
                    document=doc_content,
                    score=score,
                    rank=rank
                ))
        
        t_end = time.time()
        
        return SearchResponse(
            query=request.query,
            results=final_results,
            total_found=len(final_results),
            retrieval_time_ms=(t_end - t_start) * 1000
        )

    async def generate_answer(self, request: RAGRequest) -> RAGResponse:
        t_start = time.time()
        
        search_req = SearchRequest(
            query=request.question,
            top_k=request.top_k,
            min_score=request.min_score
        )
        search_res = await self.search(search_req)
        t_retrieval = time.time()

        answer = await self.llm.generate_answer(
            question=request.question,
            context_documents=search_res.results
        )
        t_gen = time.time()

        return RAGResponse(
            question=request.question,
            answer=answer,
            sources=search_res.results,
            retrieval_time_ms=(t_retrieval - t_start) * 1000,
            generation_time_ms=(t_gen - t_retrieval) * 1000,
            total_time_ms=(t_gen - t_start) * 1000
        )

def get_rag_service(milvus_client, es_service, embedding_service, llm_service) -> RAGService:
    return RAGService(milvus_client, es_service, embedding_service, llm_service)
