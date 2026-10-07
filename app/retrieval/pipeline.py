from typing import List, Optional

from app.config import Settings, get_settings
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.rrf import RRFCombiner
from app.retrieval.sbert import SBERTRetriever
from app.schemas.paper import ResearchDocument
from app.schemas.retrieval import RetrievedCandidate


class HybridRetrievalPipeline:
    """End-to-end hybrid retrieval: Query -> BM25 + SBERT -> RRF -> Bounded Candidate Pool -> Cross-Encoder."""

    def __init__(
        self,
        bm25: Optional[BM25Retriever] = None,
        sbert: Optional[SBERTRetriever] = None,
        rrf: Optional[RRFCombiner] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        default_candidate_limit: int = 20,
    ) -> None:
        self.bm25 = bm25 or BM25Retriever()
        self.sbert = sbert or SBERTRetriever()
        self.rrf = rrf or RRFCombiner()
        self.reranker = reranker
        self.default_candidate_limit = default_candidate_limit

    async def retrieve(
        self,
        query: str,
        documents: List[ResearchDocument],
        candidate_limit: Optional[int] = None,
        rerank: bool = True,
    ) -> List[RetrievedCandidate]:
        """Execute BM25 and SBERT retrieval, merge via RRF, optionally rerank via Cross-Encoder, and return bounded pool."""
        if not documents:
            return []

        limit = candidate_limit if candidate_limit is not None else self.default_candidate_limit
        bounded_limit = max(1, min(limit, 100))

        # 1. Lexical retrieval via BM25
        bm25_ranked = self.bm25.rank(query, documents)

        # 2. Semantic retrieval via SBERT
        sbert_ranked = self.sbert.rank(query, documents)

        # 3. Reciprocal Rank Fusion aggregation
        candidates = self.rrf.combine(
            bm25_results=bm25_ranked,
            sbert_results=sbert_ranked,
            candidate_limit=bounded_limit,
        )

        # 4. Cross-Encoder reranking (Phase 5)
        if rerank and self.reranker is not None:
            candidates = await self.reranker.rerank(
                query=query,
                candidates=candidates,
                candidate_limit=bounded_limit,
            )

        return candidates


def get_retrieval_pipeline() -> HybridRetrievalPipeline:
    """Dependency provider returning configured HybridRetrievalPipeline instance."""
    settings = get_settings()
    bm25 = BM25Retriever(k1=settings.BM25_K1, b=settings.BM25_B)
    sbert = SBERTRetriever(model_name=settings.SBERT_MODEL_NAME)
    rrf = RRFCombiner(
        k=settings.RRF_K,
        bm25_weight=settings.RRF_BM25_WEIGHT,
        sbert_weight=settings.RRF_SBERT_WEIGHT,
    )
    reranker = None
    if settings.CROSS_ENCODER_ENABLED:
        reranker = CrossEncoderReranker(
            model_name=settings.CROSS_ENCODER_MODEL_NAME,
            model_version=settings.CROSS_ENCODER_MODEL_VERSION,
            batch_size=settings.CROSS_ENCODER_BATCH_SIZE,
            device=settings.CROSS_ENCODER_DEVICE,
            candidate_limit=settings.CROSS_ENCODER_TOP_K,
        )

    return HybridRetrievalPipeline(
        bm25=bm25,
        sbert=sbert,
        rrf=rrf,
        reranker=reranker,
        default_candidate_limit=settings.RETRIEVAL_CANDIDATE_LIMIT,
    )

