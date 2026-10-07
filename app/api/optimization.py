"""
Optimization API endpoints for ResearchMind AI.

Supports:
- POST /api/v1/optimize: Solves R* = argmax_{R in S_n} U(R | q, D, F, P, C)
  with Stage 1 Linear Assignment or Stage 2 MILP Diversity, and runs
  independent verification.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from app.feedback.validator import FeedbackContradictionError
from app.optimization.constraints import ConstraintContradictionError
from app.optimization.models import OptimizationProblem, OptimizationResult
from app.optimization.optimizer import GlobalRankingOptimizer

logger = logging.getLogger("researchmind.api.optimization")

router = APIRouter(prefix="/optimize", tags=["Optimization"])


@router.post(
    "",
    response_model=OptimizationResult,
    summary="Global Ranking Optimization & Verification",
    description=(
        "Solves the formal global ranking optimization problem: "
        "R* = argmax_{R in S_n} U(R | q, D, F, P, C). "
        "Dispatches to Stage 1 Linear Assignment (Hungarian algorithm) or "
        "Stage 2 Pairwise Diversity (MILP). Performs independent verification "
        "of constraints and objective values, and records exact approximation gaps."
    ),
)
def optimize_ranking(problem: OptimizationProblem) -> OptimizationResult:
    """Execute global ranking optimization with independent verification."""
    optimizer = GlobalRankingOptimizer()
    try:
        result = optimizer.optimize(problem)
        return result
    except (FeedbackContradictionError, ConstraintContradictionError) as exc:
        logger.warning("Contradiction in optimization request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Unexpected error in ranking optimization: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(exc)}",
        )
