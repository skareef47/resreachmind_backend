from typing import Dict, List, Optional

from app.retrieval.bm25 import BM25RankedResult
from app.retrieval.sbert import SBERTRankedResult
from app.schemas.paper import ResearchDocument
from app.schemas.retrieval import RetrievalMetadata, RetrievedCandidate


class RRFCombiner:
    """Reciprocal Rank Fusion (RRF) rank aggregation combiner."""

    def __init__(
        self,
        k: int = 60,
        bm25_weight: float = 1.0,
        sbert_weight: float = 1.0,
    ) -> None:
        self.k = k
        self.bm25_weight = bm25_weight
        self.sbert_weight = sbert_weight

    def combine(
        self,
        bm25_results: List[BM25RankedResult],
        sbert_results: List[SBERTRankedResult],
        candidate_limit: Optional[int] = None,
    ) -> List[RetrievedCandidate]:
        """Combine rankings from BM25 and SBERT using RRF without raw score coupling."""
        # Index results by internal document ID
        bm25_map: Dict[str, BM25RankedResult] = {r.document.internal_id: r for r in bm25_results}
        sbert_map: Dict[str, SBERTRankedResult] = {r.document.internal_id: r for r in sbert_results}

        # Gather all unique documents
        all_docs: Dict[str, ResearchDocument] = {}
        for r in bm25_results:
            all_docs[r.document.internal_id] = r.document
        for r in sbert_results:
            all_docs[r.document.internal_id] = r.document

        if not all_docs:
            return []

        # Calculate RRF score for every candidate
        scored_candidates: List[tuple[float, ResearchDocument, RetrievalMetadata]] = []

        for doc_id, doc in all_docs.items():
            bm25_item = bm25_map.get(doc_id)
            sbert_item = sbert_map.get(doc_id)

            rrf_score = 0.0

            bm25_rank: Optional[int] = None
            bm25_score: Optional[float] = None
            if bm25_item is not None:
                bm25_rank = bm25_item.rank
                bm25_score = bm25_item.score
                rrf_score += self.bm25_weight / (self.k + bm25_rank)

            sbert_rank: Optional[int] = None
            sbert_score: Optional[float] = None
            if sbert_item is not None:
                sbert_rank = sbert_item.rank
                sbert_score = sbert_item.score
                rrf_score += self.sbert_weight / (self.k + sbert_rank)

            metadata = RetrievalMetadata(
                bm25_rank=bm25_rank,
                bm25_score=bm25_score,
                sbert_rank=sbert_rank,
                sbert_score=sbert_score,
                rrf_score=round(rrf_score, 6),
                rrf_rank=0,  # Assigned after sorting
            )
            scored_candidates.append((rrf_score, doc, metadata))

        # Sort descending by RRF score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        limit = candidate_limit if candidate_limit is not None else len(scored_candidates)
        final_candidates: List[RetrievedCandidate] = []

        for rank_idx, (_, doc, meta) in enumerate(scored_candidates[:limit], start=1):
            meta.rrf_rank = rank_idx
            final_candidates.append(
                RetrievedCandidate(
                    document=doc,
                    metadata=meta,
                )
            )

        return final_candidates
