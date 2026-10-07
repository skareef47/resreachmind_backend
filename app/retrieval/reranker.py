import logging
import math
import re
import time
from typing import Any, List, Optional, Protocol, Tuple

from app.config import get_settings
from app.retrieval.bm25 import tokenize_text
from app.retrieval.features import extract_candidate_features, sigmoid
from app.schemas.retrieval import (
    RerankerModelMetadata,
    RerankMetadata,
    RetrievedCandidate,
)

logger = logging.getLogger("researchmind.retrieval.reranker")


class CrossEncoderEngine(Protocol):
    """Interface for Cross-Encoder model inference engines."""

    def predict(self, pairs: List[Tuple[str, str]], batch_size: int = 16) -> List[float]:
        ...


class DeterministicCrossEncoderEngine:
    """
    Deterministic cross-attention relevance approximation engine.
    
    Provides fine-grained query-document cross scoring without requiring external
    model weight downloads. Evaluates token intersection, exact phrase match,
    positional weighting, and technical acronym alignment to emit calibrated logits.
    """

    def predict(self, pairs: List[Tuple[str, str]], batch_size: int = 16) -> List[float]:
        results: List[float] = []
        for query, text in pairs:
            results.append(self._score_pair(query, text))
        return results

    def _score_pair(self, query: str, text: str) -> float:
        q_tokens = tokenize_text(query)
        doc_tokens = tokenize_text(text)

        if not q_tokens or not doc_tokens:
            return -2.0

        q_lower = query.lower().strip()
        doc_lower = text.lower()

        doc_token_set = set(doc_tokens)
        matched_tokens = [t for t in q_tokens if t in doc_token_set]
        coverage = len(matched_tokens) / len(q_tokens)

        # Baseline logit from coverage
        logit = (coverage * 4.0) - 2.0

        # Exact phrase bonus
        if q_lower in doc_lower:
            logit += 2.5

        # Acronym & exact technical term bonus (e.g. BERT, RRF, ViT, GNN)
        acronyms = re.findall(r"\b[A-Z0-9]{2,}\b", query)
        for acr in acronyms:
            if acr in text:
                logit += 1.5

        # Early position discount: bonus if query terms appear in first 150 characters (title)
        first_segment = doc_lower[:150]
        title_matches = sum(1 for t in q_tokens if t in first_segment)
        if title_matches > 0:
            logit += (title_matches / len(q_tokens)) * 1.5

        return round(logit, 4)


class SentenceTransformerCrossEncoderEngine:
    """Production Cross-Encoder backend utilizing sentence-transformers."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info("Loaded CrossEncoder model '%s' on device '%s'", self.model_name, self.device)
        except Exception as exc:
            logger.warning(
                "Unable to load CrossEncoder model '%s': %s; falling back to deterministic engine",
                self.model_name,
                exc,
            )
            self._model = None

    def predict(self, pairs: List[Tuple[str, str]], batch_size: int = 16) -> List[float]:
        if self._model is not None:
            # Predict in batches
            all_scores: List[float] = []
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]
                batch_scores = self._model.predict(batch)
                if hasattr(batch_scores, "tolist"):
                    all_scores.extend(batch_scores.tolist())
                else:
                    all_scores.extend([float(s) for s in batch_scores])
            return all_scores
        else:
            return DeterministicCrossEncoderEngine().predict(pairs, batch_size=batch_size)


class FailingCrossEncoderEngine:
    """Mock engine that unconditionally raises an error to verify graceful failure."""

    def __init__(self, error_message: str = "Simulated Cross-Encoder hardware failure") -> None:
        self.error_message = error_message

    def predict(self, pairs: List[Tuple[str, str]], batch_size: int = 16) -> List[float]:
        raise RuntimeError(self.error_message)


class CrossEncoderReranker:
    """
    Cross-Encoder Reranker for the bounded Phase 4 candidate pool.
    
    Features:
    - Configurable model & device (CPU/GPU)
    - Batch inference
    - Candidate limit bounding
    - Sigmoid score normalization (intermediate relevance feature)
    - Latency measurement & execution telemetry
    - Graceful model failure handling (fallback preserves pre-rerank order)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        candidate_limit: Optional[int] = None,
        engine: Optional[Any] = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.CROSS_ENCODER_MODEL_NAME
        self.model_version = model_version or settings.CROSS_ENCODER_MODEL_VERSION
        self.batch_size = batch_size or settings.CROSS_ENCODER_BATCH_SIZE
        self.device = self._resolve_device(device or settings.CROSS_ENCODER_DEVICE)
        self.candidate_limit = candidate_limit or settings.CROSS_ENCODER_TOP_K

        if engine is not None:
            self.engine = engine
        else:
            self.engine = self._init_default_engine()

    def _resolve_device(self, requested_device: str) -> str:
        """Resolve device string (cpu, cuda, or auto)."""
        req = requested_device.lower().strip()
        if req == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return req

    def _init_default_engine(self) -> Any:
        """Initialize engine, preferring real CrossEncoder if available, otherwise deterministic."""
        try:
            import sentence_transformers  # noqa: F401

            return SentenceTransformerCrossEncoderEngine(
                model_name=self.model_name,
                device=self.device,
            )
        except ImportError:
            logger.info("sentence_transformers not installed; using DeterministicCrossEncoderEngine")
            return DeterministicCrossEncoderEngine()

    def _build_document_text(self, candidate: RetrievedCandidate) -> str:
        """Serialize candidate document into query-pairable text representation."""
        doc = candidate.document
        parts = [doc.title]
        if doc.abstract:
            parts.append(doc.abstract)
        if doc.concepts:
            parts.append(f"Concepts: {', '.join(doc.concepts[:5])}")
        return ". ".join(parts)

    async def rerank(
        self,
        query: str,
        candidates: List[RetrievedCandidate],
        candidate_limit: Optional[int] = None,
    ) -> List[RetrievedCandidate]:
        """
        Rerank candidates using Cross-Encoder scoring.
        
        Returns candidates ordered by reranker score with pre-rerank rank, post-rerank rank,
        reranker score, model metadata, and extracted intermediate features.
        """
        start_time = time.perf_counter()

        if not candidates:
            return []

        # Determine effective candidate pool limit
        limit = candidate_limit if candidate_limit is not None else self.candidate_limit
        bounded_limit = max(1, min(limit, 100))
        target_candidates = candidates[:bounded_limit]

        # Prepare (query, document_text) pairs
        pairs = [(query, self._build_document_text(c)) for c in target_candidates]

        try:
            # Batch inference execution
            raw_scores: List[float] = []
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i : i + self.batch_size]
                batch_scores = self.engine.predict(batch_pairs, batch_size=self.batch_size)
                raw_scores.extend(batch_scores)

            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 3)

            # Build model execution metadata
            model_meta = RerankerModelMetadata(
                model_name=self.model_name,
                model_version=self.model_version,
                device=self.device,
                batch_size=self.batch_size,
                latency_ms=elapsed_ms,
                is_fallback=False,
            )

            # Pair candidates with raw and normalized scores
            scored_candidates: List[Tuple[RetrievedCandidate, float, float]] = []
            for cand, raw_s in zip(target_candidates, raw_scores):
                norm_s = sigmoid(raw_s)
                scored_candidates.append((cand, raw_s, norm_s))

            # Sort descending by reranker score; break ties using Phase 4 RRF score
            scored_candidates.sort(
                key=lambda x: (x[2], x[0].metadata.rrf_score),
                reverse=True,
            )

            # Assemble reranked candidates with metadata and intermediate features
            reranked_pool: List[RetrievedCandidate] = []
            for post_idx, (cand, raw_s, norm_s) in enumerate(scored_candidates, start=1):
                cand_copy = cand.model_copy(deep=True)
                pre_rank = cand.metadata.rrf_rank

                cand_copy.rerank_metadata = RerankMetadata(
                    pre_rerank_rank=pre_rank,
                    post_rerank_rank=post_idx,
                    reranker_score=round(norm_s, 6),
                    raw_score=round(raw_s, 4),
                    model_metadata=model_meta,
                )
                cand_copy.features = extract_candidate_features(cand_copy)
                reranked_pool.append(cand_copy)

            return reranked_pool

        except Exception as exc:
            # Graceful model failure fallback
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
            logger.warning(
                "Cross-Encoder reranking failed: %s; gracefully falling back to pre-rerank RRF order",
                exc,
            )
            fallback_meta = RerankerModelMetadata(
                model_name=self.model_name,
                model_version=self.model_version,
                device=self.device,
                batch_size=self.batch_size,
                latency_ms=elapsed_ms,
                is_fallback=True,
            )

            fallback_pool: List[RetrievedCandidate] = []
            for post_idx, cand in enumerate(target_candidates, start=1):
                cand_copy = cand.model_copy(deep=True)
                pre_rank = cand.metadata.rrf_rank

                cand_copy.rerank_metadata = RerankMetadata(
                    pre_rerank_rank=pre_rank,
                    post_rerank_rank=post_idx,
                    reranker_score=0.0,
                    raw_score=None,
                    model_metadata=fallback_meta,
                )
                cand_copy.features = extract_candidate_features(cand_copy)
                fallback_pool.append(cand_copy)

            return fallback_pool
