"""
Evaluation API endpoints for ResearchMind AI (Phase 9).

Provides endpoints to run and inspect IR benchmarks:
- Precision@K
- Recall@K
- NDCG@K
- Search Success Rate
- User Satisfaction Score
- Optimization Verification Rate
- Buildability Evaluation against Ground Truth
"""

from typing import Any, Dict, List, Optional, Set
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.evaluation.metrics import (
    evaluate_buildability_accuracy,
    ndcg_at_k,
    optimization_verification_rate,
    precision_at_k,
    recall_at_k,
    search_success_rate,
    user_satisfaction_score,
)
from app.feedback.models import FeedbackItem
from app.intelligence.models import BuildabilityReport
from app.optimization.models import VerificationReport

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


class EvaluationBenchmarkRequest(BaseModel):
    retrieved_ids: List[str] = Field(..., description="Ordered list of candidate IDs returned by search")
    ground_truth_relevant_ids: List[str] = Field(..., description="Ground truth relevant document IDs")
    graded_relevance: Optional[Dict[str, float]] = Field(
        default=None, description="Optional graded relevance map (e.g. 0.0 to 3.0) for NDCG"
    )
    k: int = Field(default=5, ge=1, description="Rank cutoff K")


class EvaluationBenchmarkResponse(BaseModel):
    k: int
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float


@router.post(
    "/benchmark",
    response_model=EvaluationBenchmarkResponse,
    summary="Compute IR Evaluation Benchmark",
    description="Calculate Precision@K, Recall@K, and NDCG@K against ground truth relevance judgments.",
)
def evaluate_ranking(request: EvaluationBenchmarkRequest) -> EvaluationBenchmarkResponse:
    rel_set = set(request.ground_truth_relevant_ids)
    p_k = precision_at_k(request.retrieved_ids, rel_set, request.k)
    r_k = recall_at_k(request.retrieved_ids, rel_set, request.k)

    graded_map = request.graded_relevance
    if not graded_map:
        graded_map = {doc_id: 1.0 for doc_id in rel_set}

    ndcg_val = ndcg_at_k(request.retrieved_ids, graded_map, request.k)

    return EvaluationBenchmarkResponse(
        k=request.k,
        precision_at_k=p_k,
        recall_at_k=r_k,
        ndcg_at_k=ndcg_val,
    )
