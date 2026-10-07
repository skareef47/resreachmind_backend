from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class FeedbackType(str, Enum):
    """Supported feedback action types."""

    RELEVANT = "relevant"
    NOT_RELEVANT = "not_relevant"
    PREFER_SIMILAR = "prefer_similar"
    EXCLUDE_SIMILAR = "exclude_similar"
    PAIRWISE_PREFERENCE = "pairwise_preference"


class FeedbackItem(BaseModel):
    """A granular piece of user feedback concerning one or two research documents."""

    document_id: Optional[str] = Field(
        default=None,
        description="Target document ID (internal_id, source_id, or DOI) for pointwise or similarity feedback",
    )
    feedback_type: FeedbackType = Field(..., description="Action type for the feedback")
    weight: float = Field(default=1.0, ge=0.0, description="Confidence or magnitude weight of feedback")

    # Fields for pairwise preference (preferred_doc_id > inferior_doc_id)
    preferred_doc_id: Optional[str] = Field(
        default=None,
        description="ID of preferred document in pairwise comparison",
    )
    inferior_doc_id: Optional[str] = Field(
        default=None,
        description="ID of less preferred document in pairwise comparison",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when feedback was registered",
    )
    notes: Optional[str] = Field(default=None, description="Optional user or system notes")


class FeedbackCollection(BaseModel):
    """Collection of user feedback items for a session or query."""

    session_id: Optional[str] = Field(default=None, description="Unique session identifier")
    query: Optional[str] = Field(default=None, description="Associated search query")
    items: List[FeedbackItem] = Field(default_factory=list, description="List of feedback items")
