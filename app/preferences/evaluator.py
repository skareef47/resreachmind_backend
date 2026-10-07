import math
import re
from typing import Dict, List, Tuple
from app.preferences.models import ComputeRequirement, UserPreferences
from app.retrieval.features import compute_recency_score
from app.schemas.retrieval import RetrievedCandidate

LOW_COMPUTE_KEYWORDS = {
    "efficient", "distillation", "quantization", "pruning", "low-rank", "lora",
    "mobile", "edge", "lightweight", "tiny", "compact", "cpu", "on-device", "fast"
}
HIGH_COMPUTE_KEYWORDS = {
    "trillion", "billion parameters", "supercomputer", "cluster", "tpu pod",
    "massive scale", "frontier", "large-scale pretraining"
}
IMPLEMENTATION_KEYWORDS = {
    "implementation", "library", "framework", "github", "pytorch", "tensorflow",
    "open-source", "toolkit", "package", "reproducible", "pipeline", "system"
}


def _check_keywords(text: str, keywords: set) -> bool:
    text_lower = text.lower()
    return any(re.search(r"\b" + re.escape(kw) + r"\b", text_lower) for kw in keywords)


def evaluate_preference_match(
    candidate: RetrievedCandidate,
    preferences: UserPreferences,
) -> float:
    """
    Evaluate how well a candidate matches the user's multi-dimensional preferences.
    
    Evaluates 8 dimensions:
    1. Relevance
    2. Implementation relevance
    3. Recency
    4. Compute requirements
    5. Dataset availability
    6. Source-code availability
    7. Pretrained model availability
    8. Source quality
    
    Returns a weighted normalized score in [0.0, 1.0].
    """
    doc = candidate.document
    meta = candidate.metadata
    text_corpus = f"{doc.title} {doc.abstract or ''}"

    # 1. Relevance Dimension
    if candidate.rerank_metadata and candidate.rerank_metadata.reranker_score is not None:
        relevance_score = candidate.rerank_metadata.reranker_score
    elif candidate.features and "cross_encoder_score" in candidate.features:
        relevance_score = candidate.features["cross_encoder_score"]
    elif meta.sbert_score is not None:
        relevance_score = max(0.0, min(1.0, (meta.sbert_score + 1.0) / 2.0))
    else:
        relevance_score = 0.5

    # 2. Implementation Relevance Dimension
    impl_score = 0.0
    if doc.code_url:
        impl_score += 0.5
    if doc.methods:
        impl_score += 0.2
    if _check_keywords(text_corpus, IMPLEMENTATION_KEYWORDS):
        impl_score += 0.3
    impl_score = min(1.0, impl_score)

    # 3. Recency Dimension
    recency_score = compute_recency_score(doc.publication_year)

    # 4. Compute Requirements Dimension
    if preferences.compute_requirement == ComputeRequirement.LOW_COMPUTE:
        if _check_keywords(text_corpus, LOW_COMPUTE_KEYWORDS):
            compute_score = 1.0
        elif _check_keywords(text_corpus, HIGH_COMPUTE_KEYWORDS):
            compute_score = 0.2
        else:
            compute_score = 0.6
    elif preferences.compute_requirement == ComputeRequirement.HIGH_COMPUTE:
        if _check_keywords(text_corpus, HIGH_COMPUTE_KEYWORDS):
            compute_score = 1.0
        else:
            compute_score = 0.6
    else:  # BALANCED
        compute_score = 0.8

    # 5. Dataset Availability Dimension
    if doc.datasets:
        dataset_score = 1.0
    elif _check_keywords(text_corpus, {"dataset", "benchmark", "corpus", "annotations"}):
        dataset_score = 0.7
    else:
        dataset_score = 0.2

    # 6. Source-Code Availability Dimension
    code_score = 1.0 if (doc.code_url and doc.code_url.strip()) else 0.0

    # 7. Pretrained Model Availability Dimension
    if doc.models:
        model_score = 1.0
    elif _check_keywords(text_corpus, {"huggingface", "weights", "checkpoints", "pretrained model"}):
        model_score = 0.8
    else:
        model_score = 0.2

    # 8. Source Quality Dimension
    citation_cnt = doc.citation_count or 0
    # Log scaling: 10,000 citations -> ~1.0
    citation_norm = min(1.0, math.log10(max(1, citation_cnt + 1)) / 4.0)

    venue_norm = 0.4
    if doc.venue:
        venue_lower = doc.venue.lower()
        if preferences.preferred_venues:
            if any(pv.lower() in venue_lower for pv in preferences.preferred_venues):
                venue_norm = 1.0
        elif any(top in venue_lower for top in ["neurips", "icml", "cvpr", "iclr", "acl", "nature", "science"]):
            venue_norm = 0.9
        else:
            venue_norm = 0.7

    quality_score = 0.6 * citation_norm + 0.4 * venue_norm

    # Aggregate weighted sum
    dimensions: List[Tuple[float, float]] = [
        (relevance_score, preferences.relevance_weight),
        (impl_score, preferences.implementation_weight),
        (recency_score, preferences.recency_weight),
        (compute_score, preferences.compute_weight),
        (dataset_score, preferences.dataset_availability_weight),
        (code_score, preferences.source_code_availability_weight),
        (model_score, preferences.pretrained_model_availability_weight),
        (quality_score, preferences.source_quality_weight),
    ]

    total_weight = sum(w for _, w in dimensions)
    if total_weight <= 0:
        return 0.5

    weighted_sum = sum(score * w for score, w in dimensions)
    return round(max(0.0, min(1.0, weighted_sum / total_weight)), 4)
