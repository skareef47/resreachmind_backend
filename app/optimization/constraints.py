from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from app.feedback.models import FeedbackItem, FeedbackType
from app.schemas.retrieval import RetrievedCandidate


class ConstraintContradictionError(ValueError):
    """Raised when hard constraints contradict each other or contradict active user feedback."""

    def __init__(self, contradictions: List[str]) -> None:
        self.contradictions = contradictions
        message = "Constraint contradiction detected: " + "; ".join(contradictions)
        super().__init__(message)


class SearchConstraints(BaseModel):
    """
    Search constraints supporting hard required filters, hard exclusions,
    and soft constraints with calibrated penalties.
    """

    # 1. Hard Required Constraints (Candidate must satisfy all, else disqualified)
    require_code: bool = Field(default=False, description="Must have verified open-source code URL")
    require_open_access: bool = Field(default=False, description="Must be open-access publication")
    min_year: Optional[int] = Field(default=None, description="Earliest allowable publication year")
    max_year: Optional[int] = Field(default=None, description="Latest allowable publication year")
    min_citations: Optional[int] = Field(default=None, ge=0, description="Minimum citation threshold")
    required_venues: Optional[List[str]] = Field(default=None, description="Must be published in one of these venues")
    required_doc_ids: Optional[List[str]] = Field(default=None, description="Document IDs that must be included")

    # 2. Hard Exclusions (Candidate matching any is disqualified)
    exclude_doc_ids: Optional[List[str]] = Field(default=None, description="Document IDs to explicitly exclude")
    exclude_venues: Optional[List[str]] = Field(default=None, description="Venues to strictly exclude")
    exclude_concepts: Optional[List[str]] = Field(default=None, description="Concepts/topics to strictly exclude")

    # 3. Soft Constraints (Violations incur non-disqualifying penalty deductions)
    soft_min_year: Optional[int] = Field(default=None, description="Preferred minimum publication year")
    soft_min_citations: Optional[int] = Field(default=None, ge=0, description="Preferred minimum citations")
    soft_require_code: bool = Field(default=False, description="Prefer code availability; penalty if missing")
    soft_require_open_access: bool = Field(default=False, description="Prefer open access; penalty if missing")
    soft_preferred_venues: Optional[List[str]] = Field(default=None, description="Preferred venues; penalty if missing")

    # 4. Penalty scaling
    soft_penalty_multiplier: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Base scaling factor for soft penalties"
    )


class ConstraintEvaluationResult(BaseModel):
    """Result of evaluating constraints against a candidate research paper."""

    is_admissible: bool = Field(..., description="Whether candidate satisfies all hard constraints")
    total_penalties: float = Field(default=0.0, ge=0.0, description="Sum of soft constraint penalties")
    hard_violations: List[str] = Field(default_factory=list, description="Descriptions of hard violations")
    soft_penalties: Dict[str, float] = Field(
        default_factory=dict, description="Breakdown of applied soft penalties"
    )


def detect_constraint_contradictions(
    constraints: SearchConstraints,
    feedback_items: Optional[List[FeedbackItem]] = None,
) -> List[str]:
    """
    Detect logical contradictions within constraints and cross-contradictions
    between constraints and feedback.
    
    Checks:
    1. min_year > max_year
    2. Overlap between required_doc_ids and exclude_doc_ids
    3. Overlap between required_venues and exclude_venues
    4. Feedback cross-contradiction: required doc marked as NOT_RELEVANT or EXCLUDE_SIMILAR
    5. Feedback cross-contradiction: excluded doc marked as RELEVANT or PREFER_SIMILAR
    """
    contradictions: List[str] = []

    # 1. Year range contradiction
    if constraints.min_year is not None and constraints.max_year is not None:
        if constraints.min_year > constraints.max_year:
            contradictions.append(
                f"Contradictory year constraints: min_year ({constraints.min_year}) is greater than max_year ({constraints.max_year})"
            )

    # 2. Required vs Excluded Document IDs
    req_docs = set(constraints.required_doc_ids or [])
    ex_docs = set(constraints.exclude_doc_ids or [])
    overlap_docs = req_docs.intersection(ex_docs)
    if overlap_docs:
        contradictions.append(
            f"Contradictory document constraints: document(s) {sorted(list(overlap_docs))} present in both required_doc_ids and exclude_doc_ids"
        )

    # 3. Required vs Excluded Venues
    req_venues = {v.lower().strip() for v in (constraints.required_venues or [])}
    ex_venues = {v.lower().strip() for v in (constraints.exclude_venues or [])}
    overlap_venues = req_venues.intersection(ex_venues)
    if overlap_venues:
        contradictions.append(
            f"Contradictory venue constraints: venue(s) {sorted(list(overlap_venues))} present in both required_venues and exclude_venues"
        )

    # 4. Cross-contradictions with User Feedback
    if feedback_items:
        for item in feedback_items:
            doc_id = item.document_id.strip() if item.document_id else None
            if not doc_id:
                continue

            # If document is required by constraint, user cannot mark it NOT_RELEVANT or EXCLUDE_SIMILAR
            if doc_id in req_docs:
                if item.feedback_type in (FeedbackType.NOT_RELEVANT, FeedbackType.EXCLUDE_SIMILAR):
                    contradictions.append(
                        f"Cross-contradiction: document '{doc_id}' is in required_doc_ids constraint but marked as '{item.feedback_type.value}' in feedback"
                    )

            # If document is excluded by constraint, user cannot mark it RELEVANT or PREFER_SIMILAR
            if doc_id in ex_docs:
                if item.feedback_type in (FeedbackType.RELEVANT, FeedbackType.PREFER_SIMILAR):
                    contradictions.append(
                        f"Cross-contradiction: document '{doc_id}' is in exclude_doc_ids constraint but marked as '{item.feedback_type.value}' in feedback"
                    )

    return contradictions


def validate_constraints(
    constraints: SearchConstraints,
    feedback_items: Optional[List[FeedbackItem]] = None,
    raise_on_error: bool = True,
) -> List[str]:
    """Validate constraints and feedback compatibility. Raises ConstraintContradictionError on conflict."""
    conflicts = detect_constraint_contradictions(constraints, feedback_items)
    if conflicts and raise_on_error:
        raise ConstraintContradictionError(conflicts)
    return conflicts


class ConstraintEvaluator:
    """Evaluates candidate research papers against hard constraints, exclusions, and soft penalties."""

    def evaluate(
        self,
        candidate: RetrievedCandidate,
        constraints: SearchConstraints,
    ) -> ConstraintEvaluationResult:
        doc = candidate.document
        doc_ids = {doc.internal_id, doc.source_id}
        if doc.doi:
            doc_ids.add(doc.doi)

        hard_violations: List[str] = []
        soft_penalties: Dict[str, float] = {}

        # ----------------------------------------------------
        # 1. HARD EXCLUSIONS
        # ----------------------------------------------------
        if constraints.exclude_doc_ids:
            if any(eid in doc_ids for eid in constraints.exclude_doc_ids):
                hard_violations.append("Document explicitly in exclude_doc_ids")

        if constraints.exclude_venues and doc.venue:
            v_lower = doc.venue.lower().strip()
            if any(ev.lower().strip() in v_lower for ev in constraints.exclude_venues):
                hard_violations.append(f"Venue '{doc.venue}' matches exclude_venues")

        if constraints.exclude_concepts and doc.concepts:
            cand_concepts = {c.lower().strip() for c in doc.concepts}
            ex_concepts = {c.lower().strip() for c in constraints.exclude_concepts}
            if cand_concepts.intersection(ex_concepts):
                hard_violations.append("Candidate matches excluded concepts")

        # ----------------------------------------------------
        # 2. HARD REQUIRED CONSTRAINTS
        # ----------------------------------------------------
        if constraints.require_code:
            if not doc.code_url or not doc.code_url.strip():
                hard_violations.append("Missing required source code URL")

        if constraints.require_open_access:
            if not doc.open_access:
                hard_violations.append("Not open-access publication")

        if constraints.min_year is not None:
            if doc.publication_year is None or doc.publication_year < constraints.min_year:
                hard_violations.append(
                    f"Publication year ({doc.publication_year}) is before min_year ({constraints.min_year})"
                )

        if constraints.max_year is not None:
            if doc.publication_year is not None and doc.publication_year > constraints.max_year:
                hard_violations.append(
                    f"Publication year ({doc.publication_year}) is after max_year ({constraints.max_year})"
                )

        if constraints.min_citations is not None:
            c_cnt = doc.citation_count or 0
            if c_cnt < constraints.min_citations:
                hard_violations.append(
                    f"Citations ({c_cnt}) below min_citations ({constraints.min_citations})"
                )

        if constraints.required_venues:
            if not doc.venue:
                hard_violations.append("Missing venue; candidate requires specific venue")
            else:
                v_lower = doc.venue.lower().strip()
                if not any(rv.lower().strip() in v_lower for rv in constraints.required_venues):
                    hard_violations.append(f"Venue '{doc.venue}' not in required_venues")

        is_admissible = len(hard_violations) == 0

        # ----------------------------------------------------
        # 3. SOFT CONSTRAINTS (Penalties)
        # ----------------------------------------------------
        mult = constraints.soft_penalty_multiplier

        if constraints.soft_require_code:
            if not doc.code_url or not doc.code_url.strip():
                soft_penalties["missing_code"] = 0.5 * mult

        if constraints.soft_require_open_access:
            if not doc.open_access:
                soft_penalties["not_open_access"] = 0.3 * mult

        if constraints.soft_min_year is not None:
            if doc.publication_year is not None and doc.publication_year < constraints.soft_min_year:
                year_diff = constraints.soft_min_year - doc.publication_year
                soft_penalties["year_below_soft_min"] = min(0.5, year_diff * 0.05) * mult

        if constraints.soft_min_citations is not None:
            c_cnt = doc.citation_count or 0
            if c_cnt < constraints.soft_min_citations:
                soft_penalties["citations_below_soft_min"] = 0.4 * mult

        if constraints.soft_preferred_venues:
            if not doc.venue or not any(
                pv.lower().strip() in doc.venue.lower().strip()
                for pv in constraints.soft_preferred_venues
            ):
                soft_penalties["outside_preferred_venues"] = 0.3 * mult

        total_penalty = round(sum(soft_penalties.values()), 4)

        return ConstraintEvaluationResult(
            is_admissible=is_admissible,
            total_penalties=total_penalty,
            hard_violations=hard_violations,
            soft_penalties=soft_penalties,
        )
