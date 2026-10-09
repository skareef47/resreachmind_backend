import math
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from app.schemas.retrieval import RetrievedCandidate


def sigmoid(z: float) -> float:
    """Compute numerically stable sigmoid function to normalize raw model logits into [0, 1]."""
    clamped = max(-50.0, min(50.0, z))
    return 1.0 / (1.0 + math.exp(-clamped))


def min_max_scale(values: List[float], min_val: float = 0.0, max_val: float = 1.0) -> List[float]:
    """Scale a list of numerical values linearly into [min_val, max_val]."""
    if not values:
        return []
    v_min = min(values)
    v_max = max(values)
    if math.isclose(v_min, v_max):
        return [min_val for _ in values]
    range_span = v_max - v_min
    target_span = max_val - min_val
    return [min_val + ((v - v_min) / range_span) * target_span for v in values]


def compute_recency_score(
    year: Optional[int], current_year: int = 2026, half_life_years: float = 5.0
) -> float:
    """Compute continuous exponential decay recency factor in [0.0, 1.0]."""
    if year is None or year <= 0:
        return 0.5  # Neutral default for unknown publication year
    delta = max(0, current_year - year)
    decay = math.pow(0.5, delta / half_life_years)
    return round(max(0.0, min(1.0, decay)), 4)


class RelevanceFeatures(BaseModel):
    """
    Multi-modal intermediate relevance feature representation.
    
    IMPORTANT ARCHITECTURAL NOTE:
    In the ResearchMind AI pipeline, the Cross-Encoder score is an intermediate
    relevance feature (capturing deep token-level cross-attention), NOT the final
    ResearchMind utility function. Final utility synthesis (combining relevance,
    recency, citation velocity, code availability, and buildability) occurs in
    subsequent optimization phases.
    """

    bm25_score: float = Field(default=0.0, description="BM25 lexical retrieval score")
    bm25_rank: Optional[int] = Field(default=None, description="BM25 rank position")
    sbert_score: float = Field(default=0.0, description="SBERT cosine similarity score")
    sbert_rank: Optional[int] = Field(default=None, description="SBERT rank position")
    rrf_score: float = Field(default=0.0, description="Reciprocal Rank Fusion hybrid score")
    rrf_rank: int = Field(default=0, description="Pre-reranking RRF rank position")
    cross_encoder_score: float = Field(
        default=0.0,
        description="Normalized Cross-Encoder score (intermediate relevance feature)",
    )
    cross_encoder_raw_score: Optional[float] = Field(
        default=None, description="Raw Cross-Encoder logit before sigmoid normalization"
    )
    citation_count: int = Field(default=0, description="Scholarly citation count")
    log_citation_count: float = Field(
        default=0.0, description="log1p transformed citation count for numerical stability"
    )
    recency_score: float = Field(default=0.0, description="Exponential decay publication recency")
    has_code: float = Field(default=0.0, description="Indicator for open-source code availability (0.0 or 1.0)")
    has_open_access: float = Field(
        default=0.0, description="Indicator for open-access availability (0.0 or 1.0)"
    )
    buildability: float = Field(
        default=0.0, description="Paper-specific engineering buildability score [0.0, 1.0]"
    )
    utility_score: float = Field(
        default=0.0, description="Preliminary scalar utility score [0.0, 1.0]"
    )

    def to_dict(self) -> Dict[str, float]:
        """Convert features to a dictionary of float values for downstream modeling."""
        return {
            "bm25_score": float(self.bm25_score),
            "bm25_rank": float(self.bm25_rank or 0),
            "sbert_score": float(self.sbert_score),
            "sbert_rank": float(self.sbert_rank or 0),
            "rrf_score": float(self.rrf_score),
            "rrf_rank": float(self.rrf_rank),
            "cross_encoder_score": float(self.cross_encoder_score),
            "cross_encoder_raw_score": float(self.cross_encoder_raw_score or 0.0),
            "citation_count": float(self.citation_count),
            "log_citation_count": float(self.log_citation_count),
            "recency_score": float(self.recency_score),
            "has_code": float(self.has_code),
            "has_open_access": float(self.has_open_access),
            "buildability": float(self.buildability),
            "utility_score": float(self.utility_score),
        }


def extract_candidate_features(
    candidate: RetrievedCandidate,
    current_year: int = 2026,
) -> Dict[str, float]:
    """
    Extract intermediate relevance and metadata features for a candidate research document.
    
    Treats Cross-Encoder score as an intermediate relevance feature alongside lexical,
    semantic, and bibliographic priors, and computes paper-specific buildability evidence.
    """
    doc = candidate.document
    meta = candidate.metadata
    rerank = candidate.rerank_metadata

    citation_cnt = doc.citation_count or 0
    log_citation = round(math.log1p(max(0, citation_cnt)), 4)
    recency = compute_recency_score(doc.publication_year, current_year=current_year)
    has_code_val = 1.0 if bool(doc.code_url and doc.code_url.strip()) else 0.0
    has_oa_val = 1.0 if bool(doc.open_access) else 0.0

    ce_score = rerank.reranker_score if rerank is not None else 0.0
    ce_raw = rerank.raw_score if rerank is not None else None

    # Paper-specific buildability evaluation
    from app.intelligence.buildability import BuildabilityEvaluator
    bld_report = BuildabilityEvaluator().evaluate_document(doc)
    bld_val = round(bld_report.overall_score, 4)
    doc.buildability_score = bld_val
    doc.buildability = bld_val
    doc.buildability_breakdown = bld_report.evidence_breakdown
    doc.evidence_sources = bld_report.evidence_sources
    doc.unknown_fields = bld_report.unknown_fields

    # Preliminary utility proxy U(R) before formal constraints or optimization
    qual_norm = min(1.0, math.log10(max(1, citation_cnt + 1)) / 4.0)
    prelim_u = round(0.40 * ce_score + 0.25 * bld_val + 0.20 * qual_norm + 0.15 * recency, 6)

    features = RelevanceFeatures(
        bm25_score=meta.bm25_score or 0.0,
        bm25_rank=meta.bm25_rank,
        sbert_score=meta.sbert_score or 0.0,
        sbert_rank=meta.sbert_rank,
        rrf_score=meta.rrf_score,
        rrf_rank=meta.rrf_rank,
        cross_encoder_score=ce_score,
        cross_encoder_raw_score=ce_raw,
        citation_count=citation_cnt,
        log_citation_count=log_citation,
        recency_score=recency,
        has_code=has_code_val,
        has_open_access=has_oa_val,
        buildability=bld_val,
        utility_score=prelim_u,
    )

    return features.to_dict()

