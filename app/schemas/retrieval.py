from typing import Dict, Optional
from pydantic import BaseModel, Field

from app.schemas.paper import ResearchDocument


class RetrievalMetadata(BaseModel):
    """Ranking scores and positions across BM25, SBERT, and RRF."""

    bm25_rank: Optional[int] = Field(default=None, description="Rank in BM25 results (1-indexed)")
    bm25_score: Optional[float] = Field(default=None, description="Raw BM25 score")
    sbert_rank: Optional[int] = Field(default=None, description="Rank in SBERT semantic results (1-indexed)")
    sbert_score: Optional[float] = Field(default=None, description="Cosine similarity score from SBERT embeddings")
    rrf_score: float = Field(..., description="Reciprocal Rank Fusion combined score")
    rrf_rank: int = Field(..., description="Final hybrid retrieval rank in bounded candidate pool (1-indexed)")


class RerankerModelMetadata(BaseModel):
    """Metadata describing the Cross-Encoder model and execution telemetry."""

    model_name: str = Field(..., description="Name or path of Cross-Encoder model")
    model_version: str = Field(default="1.0.0", description="Model version or checkpoint revision")
    device: str = Field(default="cpu", description="Execution device (e.g. cpu, cuda)")
    batch_size: int = Field(default=16, description="Batch size used during inference")
    latency_ms: float = Field(default=0.0, description="Inference latency in milliseconds")
    is_fallback: bool = Field(default=False, description="Whether fallback scoring was used due to model failure")


class RerankMetadata(BaseModel):
    """Metadata for Cross-Encoder reranking."""

    pre_rerank_rank: int = Field(..., description="Candidate rank prior to reranking (from RRF, 1-indexed)")
    reranker_score: float = Field(..., description="Normalized Cross-Encoder score (intermediate relevance feature)")
    post_rerank_rank: int = Field(..., description="Candidate rank after Cross-Encoder reranking (1-indexed)")
    raw_score: Optional[float] = Field(default=None, description="Raw Cross-Encoder output logit before normalization")
    model_metadata: Optional[RerankerModelMetadata] = Field(
        default=None, description="Cross-Encoder model telemetry"
    )


class RetrievedCandidate(BaseModel):
    """A candidate research paper along with hybrid retrieval and reranking metadata."""

    document: ResearchDocument = Field(..., description="Canonical research document")
    metadata: RetrievalMetadata = Field(..., description="Detailed retrieval scores and ranks")
    rerank_metadata: Optional[RerankMetadata] = Field(
        default=None,
        description="Cross-Encoder reranking metadata (pre/post rank, score, model info)",
    )
    features: Optional[Dict[str, float]] = Field(
        default=None,
        description="Intermediate multi-modal relevance feature dict",
    )

