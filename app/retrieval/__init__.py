"""Hybrid retrieval and reranking package: BM25, SBERT, RRF, Cross-Encoder, and Features."""

from app.retrieval.bm25 import BM25RankedResult, BM25Retriever, tokenize_text
from app.retrieval.features import (
    RelevanceFeatures,
    extract_candidate_features,
    min_max_scale,
    sigmoid,
)
from app.retrieval.pipeline import HybridRetrievalPipeline, get_retrieval_pipeline
from app.retrieval.reranker import (
    CrossEncoderReranker,
    DeterministicCrossEncoderEngine,
    FailingCrossEncoderEngine,
)
from app.retrieval.rrf import RRFCombiner
from app.retrieval.sbert import SBERTRankedResult, SBERTRetriever, cosine_similarity

__all__ = [
    "BM25Retriever",
    "BM25RankedResult",
    "tokenize_text",
    "SBERTRetriever",
    "SBERTRankedResult",
    "cosine_similarity",
    "RRFCombiner",
    "CrossEncoderReranker",
    "DeterministicCrossEncoderEngine",
    "FailingCrossEncoderEngine",
    "RelevanceFeatures",
    "extract_candidate_features",
    "sigmoid",
    "min_max_scale",
    "HybridRetrievalPipeline",
    "get_retrieval_pipeline",
]

