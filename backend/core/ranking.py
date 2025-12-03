from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class FusionConfig:
    rrf_k: int = 60
    alpha: float = 0.5
    retrieval_multiplier: int = 5

class Ranker:
    @staticmethod
    def _normalize_scores(score_map: Dict[str, float]) -> Dict[str, float]:
        if not score_map: return {}
        scores = list(score_map.values())
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s: return {k: 1.0 for k in score_map}
        return {k: (v - min_s) / (max_s - min_s) for k, v in score_map.items()}

    def fuse_full_scores(
        self,
        dense_scores: Dict[str, float],
        sparse_raw_scores: Dict[str, float],
        config: FusionConfig
    ) -> List[Tuple[str, float]]:
        """
        Fuses scores where every document has both a Dense and a Sparse Raw score.
        1. Convert Sparse Raw Score -> Rank -> RRF Score.
        2. Normalize Dense and Sparse RRF.
        3. Weighted Mean.
        """
        
        # 1. Calculate RRF Score for Sparse
        # Sort IDs by raw BM25 score descending to determine rank
        sorted_by_sparse = sorted(sparse_raw_scores.items(), key=lambda x: x[1], reverse=True)
        
        sparse_rrf_map = {}
        for rank, (doc_id, _) in enumerate(sorted_by_sparse, start=1):
            # RRF Logic: 1 / (k + rank)
            sparse_rrf_map[doc_id] = 1.0 / (config.rrf_k + rank)

        # 2. Normalize
        norm_dense = self._normalize_scores(dense_scores)
        norm_sparse = self._normalize_scores(sparse_rrf_map)
        
        # 3. Weighted Sum
        final_scores = {}
        all_ids = set(norm_dense.keys())
        
        for doc_id in all_ids:
            d_val = norm_dense.get(doc_id, 0.0)
            s_val = norm_sparse.get(doc_id, 0.0)
            
            final_scores[doc_id] = (config.alpha * d_val) + ((1.0 - config.alpha) * s_val)
            
        return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    