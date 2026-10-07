"""User feedback package: pointwise, similarity, pairwise feedback, validation, and matching."""

from app.feedback.matcher import FeedbackMatcher
from app.feedback.models import FeedbackCollection, FeedbackItem, FeedbackType
from app.feedback.validator import (
    FeedbackContradictionError,
    detect_feedback_contradictions,
    validate_feedback,
)

__all__ = [
    "FeedbackType",
    "FeedbackItem",
    "FeedbackCollection",
    "FeedbackContradictionError",
    "detect_feedback_contradictions",
    "validate_feedback",
    "FeedbackMatcher",
]
