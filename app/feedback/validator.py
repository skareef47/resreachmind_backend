from typing import List, Set, Tuple

from app.feedback.models import FeedbackItem, FeedbackType


class FeedbackContradictionError(ValueError):
    """Raised when feedback contains logical contradictions that cannot be silently resolved."""

    def __init__(self, contradictions: List[str]) -> None:
        self.contradictions = contradictions
        message = "Contradictory feedback detected: " + "; ".join(contradictions)
        super().__init__(message)


def detect_feedback_contradictions(feedback_items: List[FeedbackItem]) -> List[str]:
    """
    Detect logical contradictions across feedback items.
    
    Checks for:
    1. Pointwise direct conflicts (same document marked RELEVANT and NOT_RELEVANT).
    2. Similarity direct conflicts (same document marked PREFER_SIMILAR and EXCLUDE_SIMILAR).
    3. Polarity mismatch (document marked NOT_RELEVANT and PREFER_SIMILAR, or RELEVANT and EXCLUDE_SIMILAR).
    4. Pairwise cycles and self-references (A > B and B > A, or A > A).
    5. Pairwise vs pointwise inversions (A > B asserted, but A is NOT_RELEVANT and B is RELEVANT).
    
    Returns a list of human-readable contradiction descriptions.
    """
    contradictions: List[str] = []

    relevant_docs: Set[str] = set()
    not_relevant_docs: Set[str] = set()
    prefer_similar_docs: Set[str] = set()
    exclude_similar_docs: Set[str] = set()
    pairwise_pairs: Set[Tuple[str, str]] = set()

    for item in feedback_items:
        # Validate pairwise formatting
        if item.feedback_type == FeedbackType.PAIRWISE_PREFERENCE:
            if not item.preferred_doc_id or not item.inferior_doc_id:
                contradictions.append(
                    "Pairwise preference requires both 'preferred_doc_id' and 'inferior_doc_id'"
                )
                continue

            pref = item.preferred_doc_id.strip()
            inf = item.inferior_doc_id.strip()

            if pref == inf:
                contradictions.append(
                    f"Self-referential pairwise preference: document '{pref}' cannot be preferred over itself"
                )
                continue

            # Check for direct cycle: B > A already present when A > B is seen
            if (inf, pref) in pairwise_pairs:
                contradictions.append(
                    f"Direct pairwise preference cycle: '{pref}' is preferred over '{inf}', but '{inf}' is also preferred over '{pref}'"
                )
            pairwise_pairs.add((pref, inf))
            continue

        # Pointwise & similarity validation
        doc_id = item.document_id.strip() if item.document_id else None
        if not doc_id:
            contradictions.append(
                f"Feedback of type '{item.feedback_type.value}' requires a non-empty 'document_id'"
            )
            continue

        if item.feedback_type == FeedbackType.RELEVANT:
            if doc_id in not_relevant_docs:
                contradictions.append(
                    f"Document '{doc_id}' is marked as both 'relevant' and 'not_relevant'"
                )
            if doc_id in exclude_similar_docs:
                contradictions.append(
                    f"Document '{doc_id}' is marked as 'relevant' but also 'exclude_similar'"
                )
            relevant_docs.add(doc_id)

        elif item.feedback_type == FeedbackType.NOT_RELEVANT:
            if doc_id in relevant_docs:
                contradictions.append(
                    f"Document '{doc_id}' is marked as both 'relevant' and 'not_relevant'"
                )
            if doc_id in prefer_similar_docs:
                contradictions.append(
                    f"Document '{doc_id}' is marked as 'not_relevant' but also 'prefer_similar'"
                )
            not_relevant_docs.add(doc_id)

        elif item.feedback_type == FeedbackType.PREFER_SIMILAR:
            if doc_id in exclude_similar_docs:
                contradictions.append(
                    f"Document '{doc_id}' is marked as both 'prefer_similar' and 'exclude_similar'"
                )
            if doc_id in not_relevant_docs:
                contradictions.append(
                    f"Document '{doc_id}' is marked as 'prefer_similar' but also 'not_relevant'"
                )
            prefer_similar_docs.add(doc_id)

        elif item.feedback_type == FeedbackType.EXCLUDE_SIMILAR:
            if doc_id in prefer_similar_docs:
                contradictions.append(
                    f"Document '{doc_id}' is marked as both 'prefer_similar' and 'exclude_similar'"
                )
            if doc_id in relevant_docs:
                contradictions.append(
                    f"Document '{doc_id}' is marked as 'exclude_similar' but also 'relevant'"
                )
            exclude_similar_docs.add(doc_id)

    # Check pairwise vs pointwise inversions: A > B, but A is not_relevant and B is relevant
    for pref, inf in pairwise_pairs:
        if pref in not_relevant_docs and inf in relevant_docs:
            contradictions.append(
                f"Pairwise preference contradiction: '{pref}' > '{inf}', but '{pref}' was marked 'not_relevant' while '{inf}' was marked 'relevant'"
            )

    return contradictions


def validate_feedback(feedback_items: List[FeedbackItem], raise_on_error: bool = True) -> List[str]:
    """
    Validate a list of feedback items for contradictions.
    
    Raises FeedbackContradictionError if any contradiction is discovered and raise_on_error is True.
    """
    conflicts = detect_feedback_contradictions(feedback_items)
    if conflicts and raise_on_error:
        raise FeedbackContradictionError(conflicts)
    return conflicts
