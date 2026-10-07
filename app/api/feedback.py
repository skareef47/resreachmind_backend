"""
User feedback API endpoints for ResearchMind AI (Phase 9).

Connects frontend feedback submission and persistence:
- POST /api/v1/feedback: Submit user feedback
- GET /api/v1/feedback: Retrieve active user feedback items
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repository import DatabaseRepository
from app.db.session import get_db_session
from app.feedback.models import FeedbackItem, FeedbackType
from app.feedback.validator import validate_feedback

logger = logging.getLogger("researchmind.api.feedback")

router = APIRouter(prefix="/feedback", tags=["Feedback"])


class SubmitFeedbackRequest(BaseModel):
    document_id: str = Field(..., description="Target document ID")
    feedback_type: FeedbackType = Field(..., description="Type of user feedback")
    target_document_id: Optional[str] = Field(default=None, description="Comparative document ID for pairwise")
    rationale: Optional[str] = Field(default=None, description="Optional user commentary")
    user_id: str = Field(default="default_user", description="User identifier")


@router.post(
    "",
    response_model=FeedbackItem,
    status_code=status.HTTP_201_CREATED,
    summary="Submit User Feedback",
    description="Record user relevance, similarity, or pairwise feedback and validate against contradictions.",
)
async def submit_feedback(
    request: SubmitFeedbackRequest,
    session: AsyncSession = Depends(get_db_session),
) -> FeedbackItem:
    item = FeedbackItem(
        document_id=request.document_id,
        feedback_type=request.feedback_type,
        target_document_id=request.target_document_id,
        rationale=request.rationale,
    )
    repo = DatabaseRepository(session)
    existing_items = await repo.get_feedback_items(user_id=request.user_id)

    # Validate compatibility with existing feedback
    try:
        validate_feedback(existing_items + [item], raise_on_error=True)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Contradictory feedback: {str(exc)}",
        )

    await repo.save_feedback(item, user_id=request.user_id)
    return item


@router.get(
    "",
    response_model=List[FeedbackItem],
    summary="Get User Feedback",
    description="Retrieve all active user feedback items for a given user.",
)
async def get_feedback(
    user_id: str = Query(default="default_user", description="User identifier"),
    session: AsyncSession = Depends(get_db_session),
) -> List[FeedbackItem]:
    repo = DatabaseRepository(session)
    return await repo.get_feedback_items(user_id=user_id)
