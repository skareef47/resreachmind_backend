import asyncio
import pytest
from fastapi.testclient import TestClient

from app.feedback.models import FeedbackItem, FeedbackType
from app.feedback.validator import (
    FeedbackContradictionError,
    detect_feedback_contradictions,
    validate_feedback,
)
from app.main import app
from app.optimization.constraints import (
    ConstraintContradictionError,
    ConstraintEvaluator,
    SearchConstraints,
    detect_constraint_contradictions,
    validate_constraints,
)
from app.optimization.utility import (
    FormalUtilityModel,
    UtilityWeights,
    get_utility_run_store,
)
from app.preferences.models import ComputeRequirement, UserPreferences
from app.schemas.paper import ResearchDocument
from app.schemas.retrieval import RetrievalMetadata, RetrievedCandidate

client = TestClient(app)


def _make_candidate(
    doc_id: str,
    title: str,
    abstract: str = "",
    year: int = 2024,
    citations: int = 50,
    venue: str = "NeurIPS",
    code_url: str = "",
    open_access: bool = True,
    concepts: list = None,
    methods: list = None,
    datasets: list = None,
    models: list = None,
    rrf_score: float = 0.03,
    reranker_score: float = 0.85,
) -> RetrievedCandidate:
    doc = ResearchDocument(
        internal_id=f"doc_{doc_id}",
        source="openalex",
        source_id=doc_id,
        title=title,
        abstract=abstract,
        authors=["Author A"],
        publication_year=year,
        citation_count=citations,
        venue=venue,
        code_url=code_url if code_url else None,
        open_access=open_access,
        concepts=concepts or [],
        methods=methods or [],
        datasets=datasets or [],
        models=models or [],
    )
    meta = RetrievalMetadata(
        bm25_rank=1,
        bm25_score=3.0,
        sbert_rank=1,
        sbert_score=0.8,
        rrf_score=rrf_score,
        rrf_rank=1,
    )
    cand = RetrievedCandidate(document=doc, metadata=meta)
    cand.features = {
        "cross_encoder_score": reranker_score,
        "bm25_score": 3.0,
        "sbert_score": 0.8,
        "rrf_score": rrf_score,
    }
    return cand


# ============================================================================
# 1. FEEDBACK CONTRADICTIONS & VALIDATION
# ============================================================================

def test_feedback_pointwise_contradiction() -> None:
    """Verify marking document as both relevant and not_relevant raises FeedbackContradictionError."""
    items = [
        FeedbackItem(document_id="doc_1", feedback_type=FeedbackType.RELEVANT),
        FeedbackItem(document_id="doc_1", feedback_type=FeedbackType.NOT_RELEVANT),
    ]
    conflicts = detect_feedback_contradictions(items)
    assert len(conflicts) > 0
    assert any("both 'relevant' and 'not_relevant'" in c for c in conflicts)

    with pytest.raises(FeedbackContradictionError):
        validate_feedback(items, raise_on_error=True)


def test_feedback_similarity_contradiction() -> None:
    """Verify marking document as both prefer_similar and exclude_similar raises error."""
    items = [
        FeedbackItem(document_id="doc_2", feedback_type=FeedbackType.PREFER_SIMILAR),
        FeedbackItem(document_id="doc_2", feedback_type=FeedbackType.EXCLUDE_SIMILAR),
    ]
    conflicts = detect_feedback_contradictions(items)
    assert len(conflicts) > 0
    assert any("both 'prefer_similar' and 'exclude_similar'" in c for c in conflicts)

    with pytest.raises(FeedbackContradictionError):
        validate_feedback(items, raise_on_error=True)


def test_feedback_pairwise_cycle_contradiction() -> None:
    """Verify pairwise direct cycle (A > B and B > A) raises error."""
    items = [
        FeedbackItem(
            feedback_type=FeedbackType.PAIRWISE_PREFERENCE,
            preferred_doc_id="doc_A",
            inferior_doc_id="doc_B",
        ),
        FeedbackItem(
            feedback_type=FeedbackType.PAIRWISE_PREFERENCE,
            preferred_doc_id="doc_B",
            inferior_doc_id="doc_A",
        ),
    ]
    with pytest.raises(FeedbackContradictionError):
        validate_feedback(items, raise_on_error=True)


def test_feedback_pairwise_vs_pointwise_contradiction() -> None:
    """Verify declaring A > B when A is not_relevant and B is relevant raises contradiction."""
    items = [
        FeedbackItem(document_id="doc_A", feedback_type=FeedbackType.NOT_RELEVANT),
        FeedbackItem(document_id="doc_B", feedback_type=FeedbackType.RELEVANT),
        FeedbackItem(
            feedback_type=FeedbackType.PAIRWISE_PREFERENCE,
            preferred_doc_id="doc_A",
            inferior_doc_id="doc_B",
        ),
    ]
    conflicts = detect_feedback_contradictions(items)
    assert any("Pairwise preference contradiction" in c for c in conflicts)
    with pytest.raises(FeedbackContradictionError):
        validate_feedback(items, raise_on_error=True)


def test_valid_feedback_passes() -> None:
    """Verify consistent feedback returns no contradictions."""
    items = [
        FeedbackItem(document_id="doc_1", feedback_type=FeedbackType.RELEVANT),
        FeedbackItem(document_id="doc_2", feedback_type=FeedbackType.NOT_RELEVANT),
        FeedbackItem(document_id="doc_3", feedback_type=FeedbackType.PREFER_SIMILAR),
        FeedbackItem(
            feedback_type=FeedbackType.PAIRWISE_PREFERENCE,
            preferred_doc_id="doc_1",
            inferior_doc_id="doc_2",
        ),
    ]
    conflicts = validate_feedback(items, raise_on_error=True)
    assert conflicts == []


# ============================================================================
# 2. CONSTRAINTS & CONTRADICTION DETECTION
# ============================================================================

def test_constraint_contradiction_min_max_year() -> None:
    """Verify min_year > max_year raises ConstraintContradictionError."""
    cons = SearchConstraints(min_year=2024, max_year=2020)
    with pytest.raises(ConstraintContradictionError):
        validate_constraints(cons, raise_on_error=True)


def test_constraint_contradiction_required_and_excluded_doc() -> None:
    """Verify same document in required and excluded lists raises error."""
    cons = SearchConstraints(
        required_doc_ids=["doc_1", "doc_2"],
        exclude_doc_ids=["doc_2", "doc_3"],
    )
    with pytest.raises(ConstraintContradictionError):
        validate_constraints(cons, raise_on_error=True)


def test_cross_contradiction_constraint_and_feedback() -> None:
    """Verify document required by constraint but marked not_relevant raises error."""
    cons = SearchConstraints(required_doc_ids=["doc_req"])
    fb = [FeedbackItem(document_id="doc_req", feedback_type=FeedbackType.NOT_RELEVANT)]
    with pytest.raises(ConstraintContradictionError):
        validate_constraints(cons, feedback_items=fb, raise_on_error=True)


def test_constraint_hard_filters_and_soft_penalties() -> None:
    """Verify hard constraints disqualify while soft constraints apply penalties."""
    evaluator = ConstraintEvaluator()

    cand_no_code = _make_candidate("1", "Paper Without Code", code_url="", year=2019)
    cand_with_code = _make_candidate("2", "Paper With Code", code_url="https://github.com/abc", year=2024)

    # 1. Hard required code
    hard_cons = SearchConstraints(require_code=True)
    res_no = evaluator.evaluate(cand_no_code, hard_cons)
    res_yes = evaluator.evaluate(cand_with_code, hard_cons)
    assert res_no.is_admissible is False
    assert "Missing required source code URL" in res_no.hard_violations
    assert res_yes.is_admissible is True

    # 2. Soft required code
    soft_cons = SearchConstraints(soft_require_code=True, soft_penalty_multiplier=0.2)
    soft_no = evaluator.evaluate(cand_no_code, soft_cons)
    assert soft_no.is_admissible is True
    assert soft_no.total_penalties > 0.0
    assert "missing_code" in soft_no.soft_penalties


# ============================================================================
# 3. PREFERENCES & UTILITY REPRODUCIBILITY
# ============================================================================

def test_preference_changes_impact_ranking() -> None:
    """Verify changing user preference profile alters the ranking order."""
    # Theoretical highly cited paper from 2017 without code
    cand_theory = _make_candidate(
        doc_id="theory",
        title="Theoretical Bounds on Attention Complexity",
        year=2017,
        citations=5000,
        code_url="",
        reranker_score=0.88,
    )

    # Practical newer paper from 2025 with open-source code
    cand_applied = _make_candidate(
        doc_id="applied",
        title="Efficient High-Throughput Attention Implementation on GPU",
        year=2025,
        citations=20,
        code_url="https://github.com/org/attention",
        reranker_score=0.85,
    )

    candidates = [cand_theory, cand_applied]
    utility_model = FormalUtilityModel()

    # Preference Profile 1: Heavily prioritize code and recency
    prefs_applied = UserPreferences(
        source_code_availability_weight=1.0,
        implementation_weight=1.0,
        recency_weight=1.0,
        source_quality_weight=0.1,
    )
    ranked1, run1 = utility_model.evaluate(candidates, preferences=prefs_applied)
    assert ranked1[0].document.source_id == "applied"

    # Preference Profile 2: Heavily prioritize source quality & citation prestige
    prefs_quality = UserPreferences(
        source_code_availability_weight=0.0,
        implementation_weight=0.0,
        recency_weight=0.0,
        source_quality_weight=1.0,
    )
    weights_quality = UtilityWeights(
        w1_relevance=0.25,
        w2_preference_match=0.45,
        w4_buildability=0.0,
        w5_source_quality=0.25,
        w6_recency=0.05,
    )
    ranked2, run2 = utility_model.evaluate(candidates, preferences=prefs_quality, weights=weights_quality)
    assert ranked2[0].document.source_id == "theory"


def test_utility_reproducibility_and_run_persistence() -> None:
    """Verify utility evaluation produces identical deterministic scores and persists run records."""
    store = get_utility_run_store()
    store.clear()

    cand1 = _make_candidate("1", "Vision Transformer", year=2023, citations=200, code_url="https://github.com/vit")
    cand2 = _make_candidate("2", "Diffusion Models", year=2024, citations=150, code_url="https://github.com/diff")
    candidates = [cand1, cand2]

    prefs = UserPreferences(implementation_weight=0.8, recency_weight=0.7)
    weights = UtilityWeights(w1_relevance=0.4, w2_preference_match=0.3, w4_buildability=0.3)
    fb = [FeedbackItem(document_id="doc_2", feedback_type=FeedbackType.RELEVANT)]

    model = FormalUtilityModel(run_store=store)

    # Run 1
    ranked1, record1 = model.evaluate(candidates, preferences=prefs, feedback_items=fb, weights=weights, persist=True)
    # Run 2 with identical inputs
    ranked2, record2 = model.evaluate(candidates, preferences=prefs, feedback_items=fb, weights=weights, persist=True)

    # Identical ordering and scores
    assert [c.document.source_id for c in ranked1] == [c.document.source_id for c in ranked2]
    u1 = [c.features["utility_score"] for c in ranked1]
    u2 = [c.features["utility_score"] for c in ranked2]
    assert u1 == u2

    # Audit persistence verification
    saved_record = store.get_run(record1.run_id)
    assert saved_record is not None
    assert saved_record.run_id == record1.run_id
    assert saved_record.weights.w1_relevance == 0.4
    assert len(saved_record.evaluations) == 2


# ============================================================================
# 4. API ENDPOINTS INTEGRATION
# ============================================================================

def test_api_feedback_validate_endpoint() -> None:
    """Verify POST /api/v1/feedback/validate succeeds on clean feedback and returns 400 on contradictions."""
    # Valid feedback
    valid_res = client.post(
        "/api/v1/feedback/validate",
        json={"items": [{"document_id": "doc_1", "feedback_type": "relevant"}]},
    )
    assert valid_res.status_code == 200
    assert valid_res.json()["valid"] is True

    # Contradictory feedback
    invalid_res = client.post(
        "/api/v1/feedback/validate",
        json={
            "items": [
                {"document_id": "doc_1", "feedback_type": "relevant"},
                {"document_id": "doc_1", "feedback_type": "not_relevant"},
            ]
        },
    )
    assert invalid_res.status_code == 400
    assert "Contradictory feedback detected" in str(invalid_res.json()["detail"])


def test_api_utility_evaluate_and_run_retrieval() -> None:
    """Verify POST /api/v1/utility/evaluate and GET /api/v1/utility/runs/{run_id}."""
    cand = _make_candidate("api_doc", "Language Model Alignment", year=2023, citations=50)

    eval_payload = {
        "candidates": [cand.model_dump(mode="json")],
        "preferences": {"implementation_weight": 0.5},
        "feedback": [{"document_id": "doc_api_doc", "feedback_type": "relevant"}],
    }

    res = client.post("/api/v1/utility/evaluate", json=eval_payload)
    assert res.status_code == 200
    data = res.json()
    assert "run_id" in data
    assert data["total_candidates"] == 1
    assert data["admissible_count"] == 1
    assert len(data["ranked_candidates"]) == 1

    run_id = data["run_id"]

    # Retrieve audit record
    run_res = client.get(f"/api/v1/utility/runs/{run_id}")
    assert run_res.status_code == 200
    audit_data = run_res.json()
    assert audit_data["run_id"] == run_id
    assert audit_data["total_candidates"] == 1
