import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from app.feedback.matcher import FeedbackMatcher
from app.feedback.models import FeedbackItem
from app.feedback.validator import validate_feedback
from app.optimization.constraints import (
    ConstraintEvaluator,
    SearchConstraints,
    validate_constraints,
)
from app.preferences.evaluator import evaluate_preference_match
from app.preferences.models import UserPreferences
from app.retrieval.features import compute_recency_score
from app.schemas.retrieval import RetrievedCandidate

logger = logging.getLogger("researchmind.optimization.utility")


class UtilityWeights(BaseModel):
    """
    Weights for formal multi-criteria utility function U(R).
    
    Formula:
    U(R) = w1*relevance + w2*preference_match + w3*feedback_match +
           w4*buildability + w5*source_quality + w6*recency - penalties
    """

    w1_relevance: float = Field(default=0.35, ge=0.0, description="Weight for intermediate relevance")
    w2_preference_match: float = Field(
        default=0.20, ge=0.0, description="Weight for multi-dimensional user preferences"
    )
    w3_feedback_match: float = Field(
        default=0.15, ge=0.0, description="Weight for user feedback alignment"
    )
    w4_buildability: float = Field(
        default=0.15, ge=0.0, description="Weight for engineering buildability & reproducibility"
    )
    w5_source_quality: float = Field(
        default=0.10, ge=0.0, description="Weight for venue reputation and citation impact"
    )
    w6_recency: float = Field(
        default=0.05, ge=0.0, description="Weight for publication recency"
    )


class UtilityComponentScores(BaseModel):
    """Detailed score breakdown for a single candidate paper under the utility model."""

    relevance: float = Field(..., description="Intermediate relevance feature score [0.0, 1.0]")
    preference_match: float = Field(..., description="User preference alignment score [0.0, 1.0]")
    feedback_match: float = Field(..., description="User feedback alignment score [0.0, 1.0]")
    buildability: float = Field(..., description="Engineering buildability proxy score [0.0, 1.0]")
    source_quality: float = Field(..., description="Citation impact and venue prestige [0.0, 1.0]")
    recency: float = Field(..., description="Publication recency factor [0.0, 1.0]")
    penalties: float = Field(default=0.0, ge=0.0, description="Soft constraint penalty deductions")
    utility_score: float = Field(..., description="Overall scalar utility score U(R)")
    is_admissible: bool = Field(..., description="Whether candidate passed all hard constraints")
    violations: List[str] = Field(default_factory=list, description="Hard violation descriptions if inadmissible")


class CandidateEvaluation(BaseModel):
    """Candidate evaluation summary for audit persistence."""

    document_id: str
    title: str
    scores: UtilityComponentScores


class UtilityRunRecord(BaseModel):
    """
    Immutable audit record persisting the exact inputs, weights, and candidate
    evaluations for a specific ranking run.
    """

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    weights: UtilityWeights
    preferences: UserPreferences
    feedback_items: List[FeedbackItem]
    constraints: SearchConstraints
    total_candidates: int
    admissible_count: int
    evaluations: List[CandidateEvaluation]


class UtilityRunStore:
    """In-memory persistence store for ranking run audits."""

    def __init__(self) -> None:
        self._runs: Dict[str, UtilityRunRecord] = {}

    def save_run(self, record: UtilityRunRecord) -> str:
        self._runs[record.run_id] = record
        return record.run_id

    def get_run(self, run_id: str) -> Optional[UtilityRunRecord]:
        return self._runs.get(run_id)

    def list_runs(self, limit: int = 50) -> List[UtilityRunRecord]:
        return list(self._runs.values())[-limit:]

    def clear(self) -> None:
        self._runs.clear()


# Global run store instance
_GLOBAL_RUN_STORE = UtilityRunStore()


def get_utility_run_store() -> UtilityRunStore:
    """Dependency provider for the audit run store."""
    return _GLOBAL_RUN_STORE


def compute_buildability_proxy(candidate: RetrievedCandidate) -> float:
    """
    Preliminary engineering buildability proxy score in [0.0, 1.0].
    
    Evaluates:
    - Verifiable code URL: +0.50
    - Pretrained model checkpoints: +0.20
    - Benchmark datasets: +0.20
    - Documented methods: +0.10
    """
    doc = candidate.document
    score = 0.0
    if doc.code_url and doc.code_url.strip():
        score += 0.50
    if doc.models:
        score += 0.20
    if doc.datasets:
        score += 0.20
    if doc.methods:
        score += 0.10
    return round(min(1.0, score), 4)


def compute_source_quality(candidate: RetrievedCandidate) -> float:
    """
    Computes normalized scholarly source quality in [0.0, 1.0].
    
    Combines log-scaled citations and recognized venue prominence.
    """
    doc = candidate.document
    citation_cnt = doc.citation_count or 0
    citation_norm = min(1.0, math.log10(max(1, citation_cnt + 1)) / 4.0)

    venue_norm = 0.4
    if doc.venue:
        venue_lower = doc.venue.lower()
        if any(top in venue_lower for top in ["neurips", "icml", "cvpr", "iclr", "acl", "nature", "science", "pnas"]):
            venue_norm = 1.0
        elif any(good in venue_lower for good in ["ieee", "acm", "aaai", "emnlp", "kdd", "eccv"]):
            venue_norm = 0.8
        else:
            venue_norm = 0.6

    return round(0.6 * citation_norm + 0.4 * venue_norm, 4)


class FormalUtilityModel:
    """
    Formal multi-criteria Utility Model U(R).
    
    Translates user preferences, feedback, and constraints into a mathematically
    rigorous utility score.
    """

    def __init__(
        self,
        default_weights: Optional[UtilityWeights] = None,
        run_store: Optional[UtilityRunStore] = None,
    ) -> None:
        self.weights = default_weights or UtilityWeights()
        self.run_store = run_store or get_utility_run_store()
        self.constraint_evaluator = ConstraintEvaluator()

    def evaluate_candidate(
        self,
        candidate: RetrievedCandidate,
        preferences: UserPreferences,
        feedback_matcher: FeedbackMatcher,
        feedback_items: List[FeedbackItem],
        constraints: SearchConstraints,
        weights: Optional[UtilityWeights] = None,
    ) -> UtilityComponentScores:
        """Calculate utility component breakdown and overall score for a candidate."""
        w = weights or self.weights
        doc = candidate.document

        # 1. Relevance component
        if candidate.rerank_metadata and candidate.rerank_metadata.reranker_score is not None:
            rel = candidate.rerank_metadata.reranker_score
        elif candidate.features and "cross_encoder_score" in candidate.features:
            rel = candidate.features["cross_encoder_score"]
        elif candidate.metadata.sbert_score is not None:
            rel = max(0.0, min(1.0, (candidate.metadata.sbert_score + 1.0) / 2.0))
        else:
            rel = 0.5

        # 2. Preference match component
        pref = evaluate_preference_match(candidate, preferences)

        # 3. Feedback match component
        fb = feedback_matcher.compute_feedback_match(candidate, feedback_items)

        # 4. Buildability component
        bld = compute_buildability_proxy(candidate)

        # 5. Source quality component
        qual = compute_source_quality(candidate)

        # 6. Recency component
        rec = compute_recency_score(doc.publication_year)

        # 7. Constraints & Penalties
        constraint_res = self.constraint_evaluator.evaluate(candidate, constraints)
        penalties = constraint_res.total_penalties

        # Compute U(R)
        raw_utility = (
            w.w1_relevance * rel
            + w.w2_preference_match * pref
            + w.w3_feedback_match * fb
            + w.w4_buildability * bld
            + w.w5_source_quality * qual
            + w.w6_recency * rec
            - penalties
        )
        utility_score = round(raw_utility, 6)

        return UtilityComponentScores(
            relevance=round(rel, 4),
            preference_match=round(pref, 4),
            feedback_match=round(fb, 4),
            buildability=round(bld, 4),
            source_quality=round(qual, 4),
            recency=round(rec, 4),
            penalties=round(penalties, 4),
            utility_score=utility_score,
            is_admissible=constraint_res.is_admissible,
            violations=constraint_res.hard_violations,
        )

    def evaluate(
        self,
        candidates: List[RetrievedCandidate],
        preferences: Optional[UserPreferences] = None,
        feedback_items: Optional[List[FeedbackItem]] = None,
        constraints: Optional[SearchConstraints] = None,
        weights: Optional[UtilityWeights] = None,
        persist: bool = True,
    ) -> Tuple[List[RetrievedCandidate], UtilityRunRecord]:
        """
        Evaluate and rank candidates under the formal utility model.
        
        Strictly detects feedback and constraint contradictions without silently
        resolving them.
        Persists exact run telemetry and weights.
        """
        prefs = preferences or UserPreferences()
        fb_list = feedback_items or []
        cons = constraints or SearchConstraints()
        w = weights or self.weights

        # 1. Contradiction Validation (DO NOT silently resolve contradictory feedback!)
        validate_feedback(fb_list, raise_on_error=True)
        validate_constraints(cons, feedback_items=fb_list, raise_on_error=True)

        # 2. Build FeedbackMatcher with corpus references
        reference_corpus = {c.document.internal_id: c.document for c in candidates}
        for c in candidates:
            if c.document.source_id:
                reference_corpus[c.document.source_id] = c.document
            if c.document.doi:
                reference_corpus[c.document.doi] = c.document

        fb_matcher = FeedbackMatcher(reference_corpus=reference_corpus)

        # 3. Evaluate each candidate
        evaluations: List[CandidateEvaluation] = []
        admissible_candidates: List[Tuple[RetrievedCandidate, UtilityComponentScores]] = []

        for cand in candidates:
            scores = self.evaluate_candidate(
                candidate=cand,
                preferences=prefs,
                feedback_matcher=fb_matcher,
                feedback_items=fb_list,
                constraints=cons,
                weights=w,
            )

            evaluations.append(
                CandidateEvaluation(
                    document_id=cand.document.internal_id,
                    title=cand.document.title,
                    scores=scores,
                )
            )

            if scores.is_admissible:
                # Attach utility score to features dictionary for downstream inspection
                cand_copy = cand.model_copy(deep=True)
                if cand_copy.features is None:
                    cand_copy.features = {}
                cand_copy.features["utility_score"] = scores.utility_score
                cand_copy.features["preference_match"] = scores.preference_match
                cand_copy.features["feedback_match"] = scores.feedback_match
                cand_copy.features["buildability"] = scores.buildability
                cand_copy.features["source_quality"] = scores.source_quality
                cand_copy.features["penalties"] = scores.penalties
                admissible_candidates.append((cand_copy, scores))

        # 4. Sort admissible candidates by utility score descending
        admissible_candidates.sort(key=lambda x: x[1].utility_score, reverse=True)
        ranked_candidates = [c for c, _ in admissible_candidates]

        # 5. Persist run record
        run_record = UtilityRunRecord(
            weights=w,
            preferences=prefs,
            feedback_items=fb_list,
            constraints=cons,
            total_candidates=len(candidates),
            admissible_count=len(ranked_candidates),
            evaluations=evaluations,
        )

        if persist:
            self.run_store.save_run(run_record)
            logger.info("Persisted ranking run '%s' with %d candidates", run_record.run_id, len(ranked_candidates))

        return ranked_candidates, run_record
