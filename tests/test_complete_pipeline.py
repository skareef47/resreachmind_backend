import math
import pytest
from fastapi.testclient import TestClient

from app.feedback.models import FeedbackItem, FeedbackType
from app.intelligence.buildability import BuildabilityEvaluator
from app.main import app
from app.optimization.constraints import SearchConstraints
from app.optimization.models import ObjectiveType, OptimizationProblem, SolverStatus, SolverType
from app.optimization.optimizer import GlobalRankingOptimizer
from app.optimization.solvers import (
    ExhaustivePermutationSolver,
    LinearAssignmentSolver,
    calculate_position_discounts,
    evaluate_ranking_objective,
)
from app.optimization.utility import FormalUtilityModel, UtilityWeights
from app.optimization.verification import OptimizationVerifier
from app.preferences.models import UserPreferences
from app.schemas.paper import ResearchDocument
from app.schemas.retrieval import RetrievalMetadata, RetrievedCandidate

client = TestClient(app)


def make_synthetic_candidate(
    doc_id: str,
    title: str,
    abstract: str = "",
    year: int = 2024,
    citations: int = 50,
    venue: str = "NeurIPS",
    code_url: str = "",
    open_access: bool = True,
    datasets: list = None,
    models: list = None,
    cross_encoder_score: float = 0.80,
) -> RetrievedCandidate:
    doc = ResearchDocument(
        internal_id=f"doc_{doc_id}",
        source="semantic_scholar",
        source_id=doc_id,
        title=title,
        abstract=abstract,
        authors=["Alice Smith", "Bob Jones"],
        publication_year=year,
        citation_count=citations,
        venue=venue,
        code_url=code_url if code_url else None,
        open_access=open_access,
        datasets=datasets or [],
        models=models or [],
    )
    bld_rep = BuildabilityEvaluator().evaluate_document(doc)
    doc.buildability_score = bld_rep.overall_score
    doc.buildability = bld_rep.overall_score
    doc.buildability_breakdown = bld_rep.evidence_breakdown
    doc.evidence_sources = bld_rep.evidence_sources
    doc.unknown_fields = bld_rep.unknown_fields

    meta = RetrievalMetadata(
        bm25_rank=1,
        bm25_score=3.5,
        sbert_rank=1,
        sbert_score=0.85,
        rrf_score=0.035,
        rrf_rank=1,
    )
    cand = RetrievedCandidate(document=doc, metadata=meta)
    cand.features = {
        "cross_encoder_score": cross_encoder_score,
        "bm25_score": 3.5,
        "sbert_score": 0.85,
        "rrf_score": 0.035,
        "buildability": bld_rep.overall_score,
    }
    return cand


# =========================================================================
# REQUIREMENT 14: IMPORTANT VALIDATION TEST (5 CANDIDATES, 6 STEPS)
# =========================================================================

def test_deterministic_5_candidate_validation_suite(capsys):
    """
    Deterministic synthetic test with 5 candidates across 6 steps:
    1. baseline ranking
    2. preference-modified ranking
    3. feedback-modified ranking
    4. constrained ranking
    5. optimizer ranking
    6. exhaustive enumeration
    Verify:
    ResearchMind optimizer utility == exhaustive optimum utility
    """
    print("\n" + "=" * 70)
    print("RESEARCHMIND 5-CANDIDATE DETERMINISTIC OPTIMIZATION VALIDATION")
    print("=" * 70)

    # Instantiate 5 distinct synthetic candidates
    c1 = make_synthetic_candidate(
        "c1",
        "Attention and Memory in Deep Transformers",
        "Theoretical analysis of memory attention.",
        year=2018,
        citations=400,
        venue="ICML",
        code_url="",
        open_access=False,
        cross_encoder_score=0.92,
    )
    c2 = make_synthetic_candidate(
        "c2",
        "Reproducible PyTorch Library for Attention Scaling",
        "Open source library with verified checkpoints and benchmarks.",
        year=2024,
        citations=45,
        venue="NeurIPS",
        code_url="https://github.com/reproducible/transformer-scaling",
        open_access=True,
        datasets=["GLUE Benchmark"],
        cross_encoder_score=0.75,
    )
    c3 = make_synthetic_candidate(
        "c3",
        "Large-Scale Benchmark Dataset for Attention",
        "Curated dataset and evaluation harness for transformers.",
        year=2022,
        citations=120,
        venue="ACL",
        code_url="",
        open_access=True,
        datasets=["AttentionCorpus-10M"],
        cross_encoder_score=0.85,
    )
    c4 = make_synthetic_candidate(
        "c4",
        "Foundational Theory of Linear Attention",
        "Mathematical proof of linear time bounds in attention mechanisms.",
        year=2023,
        citations=1500,
        venue="Nature Machine Intelligence",
        code_url="",
        open_access=False,
        cross_encoder_score=0.60,
    )
    c5 = make_synthetic_candidate(
        "c5",
        "Legacy RNN Baselines on Text Sequences",
        "Older comparison of recurrent units.",
        year=2012,
        citations=15,
        venue="Workshop",
        code_url="",
        open_access=False,
        cross_encoder_score=0.35,
    )

    candidates = [c1, c2, c3, c4, c5]
    utility_model = FormalUtilityModel()

    # Step 1: Baseline ranking (relevance prioritized baseline)
    pref_baseline = UserPreferences(
        relevance_weight=1.0,
        implementation_weight=0.0,
        recency_weight=0.0,
        source_code_availability_weight=0.0,
        dataset_availability_weight=0.0,
        pretrained_model_availability_weight=0.0,
        source_quality_weight=0.0,
    )
    ranked_baseline, _ = utility_model.evaluate(
        candidates=candidates,
        preferences=pref_baseline,
        persist=False,
    )
    baseline_order = [c.document.internal_id for c in ranked_baseline]
    print(f"Step 1 - Baseline Ranking: {baseline_order}")
    assert baseline_order[0] == "doc_c1"

    # Step 2: Preference-modified ranking (Prioritize implementation & recency)
    pref_impl = UserPreferences(
        relevance_weight=0.3,
        implementation_weight=1.0,
        recency_weight=1.0,
        source_code_availability_weight=1.0,
        dataset_availability_weight=0.8,
    )
    ranked_pref, _ = utility_model.evaluate(
        candidates=candidates,
        preferences=pref_impl,
        persist=False,
    )
    pref_order = [c.document.internal_id for c in ranked_pref]
    print(f"Step 2 - Preference-Modified Ranking: {pref_order}")
    # c2 (verified code + 2024) receives higher preference boost and rises to #1
    assert pref_order.index("doc_c2") < baseline_order.index("doc_c2")
    assert pref_order[0] == "doc_c2"

    # Step 3: Feedback-modified ranking (Endorse c3, reject c1)
    fb = [
        FeedbackItem(document_id="doc_c3", feedback_type=FeedbackType.RELEVANT, weight=1.0),
        FeedbackItem(document_id="doc_c1", feedback_type=FeedbackType.NOT_RELEVANT, weight=1.0),
    ]
    ranked_fb, _ = utility_model.evaluate(
        candidates=candidates,
        feedback_items=fb,
        persist=False,
    )
    fb_order = [c.document.internal_id for c in ranked_fb]
    print(f"Step 3 - Feedback-Modified Ranking: {fb_order}")
    # c3 rises relative to c1
    assert fb_order.index("doc_c3") < fb_order.index("doc_c1")

    # Step 4: Constrained ranking (Require verified code)
    cons = SearchConstraints(require_code=True)
    ranked_cons, run_cons = utility_model.evaluate(
        candidates=candidates,
        constraints=cons,
        persist=False,
    )
    cons_order = [c.document.internal_id for c in ranked_cons]
    print(f"Step 4 - Constrained Ranking (require_code=True): {cons_order}")
    # Only c2 satisfies require_code
    assert len(cons_order) == 1
    assert cons_order[0] == "doc_c2"

    # Step 5: Optimizer ranking (LinearAssignmentSolver / Hungarian)
    k = 5
    problem = OptimizationProblem(
        candidates=candidates,
        k=k,
        objective_type=ObjectiveType.LINEAR_POSITIONAL,
        preferences=pref_impl,
        feedback_items=fb,
    )
    optimizer = GlobalRankingOptimizer(utility_model=utility_model)
    opt_result = optimizer.optimize(problem)
    opt_order = [c.document.internal_id for c in opt_result.ranked_candidates]
    opt_utility = opt_result.objective_value
    print(f"Step 5 - Optimizer Ranking: {opt_order}")
    print(f"         Optimizer Reported Utility U(R*): {opt_utility:.6f}")
    assert opt_result.solver_status == SolverStatus.OPTIMAL

    # Step 6: Exhaustive Enumeration
    # The verifier in optimizer performed exhaustive enumeration because len(candidates) = 5 <= 10
    assert opt_result.verification.exhaustive_comparison_performed is True
    ex_utility = opt_result.verification.exact_reference_objective
    optimality_gap = opt_result.verification.approximation_gap
    print(f"Step 6 - Exhaustive Global Optimum Utility: {ex_utility:.6f}")
    print(f"         Optimality Gap (|U_opt - U_exact|): {optimality_gap:.8f}")
    print("=" * 70 + "\n")

    # VERIFY: ResearchMind optimizer utility == exhaustive optimum utility
    assert optimality_gap < 1e-4, f"Optimality gap {optimality_gap} exceeds tolerance 1e-4!"
    assert opt_result.verification.is_verified is True


# =========================================================================
# REQUIREMENT 13: SPECIFIC FEATURE TESTS (A - K)
# =========================================================================

def test_feature_a_preference_changes_ranking():
    """A. User preferences actually change candidate ranking."""
    c_theory = make_synthetic_candidate("p1", "Theoretical Machine Learning", year=2019, cross_encoder_score=0.88, code_url="")
    c_code = make_synthetic_candidate("p2", "Fast Inference Framework", year=2024, cross_encoder_score=0.76, code_url="https://github.com/fast/inf")

    model = FormalUtilityModel()
    # Baseline: High relevance priority
    pref_base = UserPreferences(
        relevance_weight=1.0,
        implementation_weight=0.0,
        source_code_availability_weight=0.0,
        recency_weight=0.0,
    )
    r_base, _ = model.evaluate([c_theory, c_code], preferences=pref_base, persist=False)
    assert r_base[0].document.internal_id == "doc_p1"

    # User toggles preference for implementation and recency
    pref_code = UserPreferences(
        relevance_weight=0.2,
        implementation_weight=1.0,
        source_code_availability_weight=1.0,
        recency_weight=0.8,
    )
    r_pref, _ = model.evaluate([c_theory, c_code], preferences=pref_code, persist=False)
    assert r_pref[0].document.internal_id == "doc_p2"


def test_feature_b_positive_feedback_changes_ranking():
    """B. Positive feedback increases candidate utility and changes ranking."""
    c1 = make_synthetic_candidate("f1", "Paper One", cross_encoder_score=0.85)
    c2 = make_synthetic_candidate("f2", "Paper Two", cross_encoder_score=0.82)

    model = FormalUtilityModel()
    r_base, _ = model.evaluate([c1, c2], persist=False)
    assert r_base[0].document.internal_id == "doc_f1"

    # Give positive feedback to Paper Two
    fb = [FeedbackItem(document_id="doc_f2", feedback_type=FeedbackType.RELEVANT, weight=1.0)]
    r_fb, _ = model.evaluate([c1, c2], feedback_items=fb, persist=False)
    assert r_fb[0].document.internal_id == "doc_f2"


def test_feature_c_negative_feedback_changes_ranking():
    """C. Negative feedback decreases candidate utility and drops ranking."""
    c1 = make_synthetic_candidate("n1", "Paper Alpha", cross_encoder_score=0.88)
    c2 = make_synthetic_candidate("n2", "Paper Beta", cross_encoder_score=0.84)

    model = FormalUtilityModel()
    r_base, _ = model.evaluate([c1, c2], persist=False)
    assert r_base[0].document.internal_id == "doc_n1"

    # Reject Paper Alpha
    fb = [FeedbackItem(document_id="doc_n1", feedback_type=FeedbackType.NOT_RELEVANT, weight=1.0)]
    r_fb, _ = model.evaluate([c1, c2], feedback_items=fb, persist=False)
    assert r_fb[0].document.internal_id == "doc_n2"


def test_feature_d_constraints_remove_penalize_candidates():
    """D. Hard constraints filter candidates; soft constraints deduct penalties."""
    c_open = make_synthetic_candidate("c_oa", "Open Access Study", open_access=True, year=2023)
    c_closed = make_synthetic_candidate("c_pay", "Paywalled Study", open_access=False, year=2023)

    model = FormalUtilityModel()
    # Hard constraint: require_open_access
    r_hard, _ = model.evaluate([c_open, c_closed], constraints=SearchConstraints(require_open_access=True), persist=False)
    assert len(r_hard) == 1
    assert r_hard[0].document.internal_id == "doc_c_oa"

    # Soft constraint: soft_require_open_access
    r_soft, _ = model.evaluate([c_open, c_closed], constraints=SearchConstraints(soft_require_open_access=True, soft_penalty_multiplier=0.2), persist=False)
    assert len(r_soft) == 2
    # c_closed has penalties applied
    closed_item = next(c for c in r_soft if c.document.internal_id == "doc_c_pay")
    assert closed_item.features["penalties"] > 0.0


def test_feature_e_different_papers_receive_different_buildability_scores():
    """E. Distinct papers receive paper-specific, evidence-grounded buildability scores."""
    c_full = make_synthetic_candidate("b1", "Full Stack AI Paper", code_url="https://github.com/ai/full", open_access=True, datasets=["MNIST"], models=["BERT"])
    c_empty = make_synthetic_candidate("b2", "Theoretical Notes", code_url="", open_access=False)

    evaluator = BuildabilityEvaluator()
    rep_full = evaluator.evaluate_document(c_full.document)
    rep_empty = evaluator.evaluate_document(c_empty.document)

    assert rep_full.overall_score != rep_empty.overall_score
    assert rep_full.overall_score > rep_empty.overall_score
    assert rep_full.evidence_breakdown["code"]["status"] == "verified"
    assert rep_empty.evidence_breakdown["code"]["status"] in ("unknown", "absent")


def test_feature_f_utility_changes_when_inputs_change():
    """F. Utility scores update dynamically when weights or preferences vary."""
    c1 = make_synthetic_candidate("u1", "AI Paper", cross_encoder_score=0.85, year=2024)

    model = FormalUtilityModel()
    _, run1 = model.evaluate([c1], weights=UtilityWeights(w1_relevance=0.8, w2_preference_match=0.1), persist=False)
    _, run2 = model.evaluate([c1], weights=UtilityWeights(w1_relevance=0.1, w2_preference_match=0.8), persist=False)

    score1 = run1.evaluations[0].scores.utility_score
    score2 = run2.evaluations[0].scores.utility_score
    assert score1 != score2


def test_feature_g_optimizer_produces_valid_ranking():
    """G. Global ranking optimizer produces permutation of correct length without duplicates."""
    cands = [make_synthetic_candidate(f"g_{i}", f"Title {i}", cross_encoder_score=0.5 + i * 0.1) for i in range(4)]
    problem = OptimizationProblem(candidates=cands, k=3)
    optimizer = GlobalRankingOptimizer()
    result = optimizer.optimize(problem)

    assert len(result.ranked_candidates) == 3
    doc_ids = [c.document.internal_id for c in result.ranked_candidates]
    assert len(set(doc_ids)) == 3
    assert result.solver_status == SolverStatus.OPTIMAL


def test_feature_h_exhaustive_verification_confirms_optimum_for_small_n():
    """H. Exhaustive verification confirms exact optimum on small instance."""
    cands = [make_synthetic_candidate(f"h_{i}", f"Paper {i}", cross_encoder_score=0.4 + i * 0.15) for i in range(4)]
    problem = OptimizationProblem(candidates=cands, k=3)
    optimizer = GlobalRankingOptimizer(verifier=OptimizationVerifier(exhaustive_threshold=5))
    res = optimizer.optimize(problem)

    assert res.verification.exhaustive_comparison_performed is True
    assert res.verification.approximation_gap is not None
    assert res.verification.approximation_gap <= 1e-4


def test_feature_i_optimality_gap_calculation():
    """I. Optimality gap is mathematically non-negative and correctly recorded."""
    cands = [make_synthetic_candidate(f"i_{i}", f"Study {i}", cross_encoder_score=0.7) for i in range(3)]
    problem = OptimizationProblem(candidates=cands, k=2)
    res = GlobalRankingOptimizer().optimize(problem)

    assert res.verification.approximation_gap is not None
    assert res.verification.approximation_gap >= 0.0


def test_feature_j_api_returns_complete_ranking_structure():
    """J. API returns complete enriched paper structure and optimization telemetry."""
    payload = {
        "query": "transformer attention mechanism",
        "top_k": 3,
        "optimize": True,
        "verification": True,
    }
    response = client.post("/api/v1/search", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "results" in data
    assert "optimization" in data
    opt = data["optimization"]
    assert opt is not None
    assert "method" in opt
    assert "status" in opt
    assert "optimality_gap" in opt
    assert "verified" in opt
    assert "guarantee_scope" in opt

    if data["results"]:
        paper = data["results"][0]
        assert "paper_id" in paper or "internal_id" in paper
        assert "relevance_score" in paper
        assert "utility_score" in paper
        assert "buildability_score" in paper
        assert "rank" in paper
        assert "explanation" in paper


def test_feature_k_frontend_consumes_response_structure():
    """K. Verify JSON contract matches frontend expected properties."""
    payload = {
        "query": "attention mechanisms",
        "limit": 3,
        "optimize": True,
    }
    res = client.post("/api/v1/search", json=payload)
    assert res.status_code == 200
    data = res.json()

    for paper in data["results"]:
        # Frontend requires title, authors, buildability_score, relevance_score, utility_score
        assert "title" in paper
        assert "source" in paper
        assert paper.get("utility_score") is not None or "utility_score" in paper.get("features", {})
        assert paper.get("relevance_score") is not None or "cross_encoder_score" in paper.get("features", {})
        assert paper.get("buildability_score") is not None
