import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.feedback.models import FeedbackItem
from app.feedback.validator import (
    FeedbackContradictionError,
    detect_feedback_contradictions,
    validate_feedback,
)
from app.optimization.constraints import (
    ConstraintContradictionError,
    SearchConstraints,
    validate_constraints,
)
from app.optimization.utility import (
    FormalUtilityModel,
    UtilityRunRecord,
    UtilityWeights,
    get_utility_run_store,
)
from app.preferences.models import UserPreferences
from app.schemas.retrieval import RetrievedCandidate

logger = logging.getLogger("researchmind.api.utility")

router = APIRouter(tags=["Feedback & Utility"])


class ValidateFeedbackRequest(BaseModel):
    items: List[FeedbackItem] = Field(..., description="List of user feedback items to validate")


class ValidateFeedbackResponse(BaseModel):
    valid: bool = Field(..., description="Whether feedback contains zero contradictions")
    contradictions: List[str] = Field(default_factory=list, description="Contradiction descriptions if any")


class EvaluateUtilityRequest(BaseModel):
    candidates: List[RetrievedCandidate] = Field(..., description="Input candidate research documents")
    preferences: Optional[UserPreferences] = Field(default=None, description="Multi-dimensional user preferences")
    feedback: Optional[List[FeedbackItem]] = Field(default=None, description="User feedback items")
    constraints: Optional[SearchConstraints] = Field(default=None, description="Hard constraints and soft penalties")
    weights: Optional[UtilityWeights] = Field(default=None, description="Utility function weights")


class EvaluateUtilityResponse(BaseModel):
    run_id: str = Field(..., description="Persisted audit run identifier")
    total_candidates: int = Field(..., description="Count of input candidates evaluated")
    admissible_count: int = Field(..., description="Count of candidates passing hard constraints")
    ranked_candidates: List[RetrievedCandidate] = Field(..., description="Candidates sorted by utility U(R) descending")
    audit_record: UtilityRunRecord = Field(..., description="Full persisted audit record")


@router.post(
    "/feedback/validate",
    response_model=ValidateFeedbackResponse,
    summary="Validate User Feedback",
    description="Check user feedback items for logical contradictions without silently resolving them.",
)
async def validate_user_feedback(request: ValidateFeedbackRequest) -> ValidateFeedbackResponse:
    contradictions = detect_feedback_contradictions(request.items)
    if contradictions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Contradictory feedback detected",
                "contradictions": contradictions,
            },
        )
    return ValidateFeedbackResponse(valid=True, contradictions=[])


@router.post(
    "/utility/evaluate",
    response_model=EvaluateUtilityResponse,
    summary="Evaluate Formal Utility Model",
    description="Calculate U(R) multi-criteria utility, filter inadmissible candidates, and persist exact run inputs.",
)
async def evaluate_utility_model(request: EvaluateUtilityRequest) -> EvaluateUtilityResponse:
    utility_model = FormalUtilityModel()

    try:
        ranked, run_record = utility_model.evaluate(
            candidates=request.candidates,
            preferences=request.preferences,
            feedback_items=request.feedback,
            constraints=request.constraints,
            weights=request.weights,
            persist=True,
        )
    except (FeedbackContradictionError, ConstraintContradictionError) as exc:
        logger.warning("Contradiction rejected in utility evaluation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    return EvaluateUtilityResponse(
        run_id=run_record.run_id,
        total_candidates=run_record.total_candidates,
        admissible_count=run_record.admissible_count,
        ranked_candidates=ranked,
        audit_record=run_record,
    )


@router.get(
    "/utility/runs/{run_id}",
    response_model=UtilityRunRecord,
    summary="Get Persisted Utility Run Record",
    description="Retrieve the exact persisted inputs, weights, and candidate evaluations for a specific ranking run.",
)
async def get_utility_run(run_id: str) -> UtilityRunRecord:
    store = get_utility_run_store()
    record = store.get_run(run_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Utility run with ID '{run_id}' not found.",
        )
    return record
