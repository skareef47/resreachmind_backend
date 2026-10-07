"""
Comprehensive tests for Phase 7 — Global Optimization and Verification.

Validates:
1. Linear assignment solver (Stage 1) with scipy.optimize.linear_sum_assignment
2. Pairwise diversity ILP solver (Stage 2) with scipy.optimize.milp
3. Richer constraints (required document inclusion) via MILP
4. Exhaustive permutation enumeration ground-truth verification (N <= 10)
5. Approximation gap recording Delta U = U(R_exact) - U(R_approx) >= 0.0
6. Independent objective re-evaluation and constraint audit
7. GlobalRankingOptimizer pipeline with formal epistemic guarantee boundary
8. Contradiction preservation
9. REST API endpoints /api/v1/optimize and search with optimize=True
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.feedback.models import FeedbackItem, FeedbackType
from app.optimization.constraints import ConstraintContradictionError, SearchConstraints
from app.optimization.models import (
    ObjectiveType,
    OptimizationProblem,
    OptimizationResult,
    SolverStatus,
    SolverType,
    VerificationStatus,
)
from app.optimization.optimizer import (
    OPTIMALITY_GUARANTEE_STATEMENT,
    GlobalRankingOptimizer,
)
from app.optimization.solvers import (
    ExhaustivePermutationSolver,
    GreedyDiversitySolver,
    LinearAssignmentSolver,
    PairwiseDiversityILPSolver,
    calculate_position_discounts,
    compute_candidate_similarity_matrix,
    evaluate_ranking_objective,
)
from app.optimization.utility import UtilityWeights
from app.optimization.verification import OptimizationVerifier
from app.preferences.models import UserPreferences
from app.schemas.paper import ResearchDocument
from app.schemas.retrieval import RetrievalMetadata, RetrievedCandidate


client = TestClient(app)


def make_candidate(
    doc_id: str,
    title: str,
    year: int = 2023,
    citations: int = 100,
    venue: str = "NeurIPS",
    concepts: list = None,
    code_url: str = "https://github.com/test/repo",
    open_access: bool = True,
    sbert_score: float = 0.8,
) -> RetrievedCandidate:
    doc = ResearchDocument(
        internal_id=doc_id,
        source="synthetic",
        source_id=doc_id,
        title=title,
        abstract=f"Abstract for {title}",
        authors=["Author A"],
        publication_year=year,
        citation_count=citations,
        venue=venue,
        concepts=concepts or ["machine learning"],
        code_url=code_url,
        open_access=open_access,
    )
    meta = RetrievalMetadata(
        bm25_rank=1,
        bm25_score=10.0,
        sbert_rank=1,
        sbert_score=sbert_score,
        rrf_rank=1,
        rrf_score=0.03,
    )
    return RetrievedCandidate(
        document=doc,
        retrieval_score=0.8,
        retrieval_rank=1,
        metadata=meta,
    )


# -------------------------------------------------------------------------
# 1. Discount Factor & Objective Mathematical Verification
# -------------------------------------------------------------------------

def test_position_discounts():
    discounts = calculate_position_discounts(4)
    assert len(discounts) == 4
    assert np.isclose(discounts[0], 1.0)                      # 1 / log2(2) = 1.0
    assert np.isclose(discounts[1], 1.0 / np.log2(3.0))       # ~ 0.6309
    assert np.isclose(discounts[2], 0.5)                      # 1 / log2(4) = 0.5
    assert np.isclose(discounts[3], 1.0 / np.log2(5.0))       # ~ 0.4307


def test_evaluate_ranking_objective_linear_and_diversity():
    utils = [0.9, 0.8, 0.4]
    discounts = calculate_position_discounts(2)  # [1.0, 0.6309]

    # Linear objective
    linear_val = evaluate_ranking_objective(
        ranking_indices=[0, 1],
        utilities=utils,
        discounts=discounts,
        objective_type=ObjectiveType.LINEAR_POSITIONAL,
    )
    expected_linear = 0.9 * 1.0 + 0.8 * (1.0 / np.log2(3.0))
    assert np.isclose(linear_val, expected_linear, atol=1e-4)

    # Diversity objective with redundancy penalty
    sim_matrix = np.array([
        [1.0, 0.9, 0.1],
        [0.9, 1.0, 0.1],
        [0.1, 0.1, 1.0],
    ])
    div_val = evaluate_ranking_objective(
        ranking_indices=[0, 1],
        utilities=utils,
        discounts=discounts,
        objective_type=ObjectiveType.PAIRWISE_DIVERSITY,
        similarity_matrix=sim_matrix,
        diversity_lambda=0.5,
    )
    expected_penalty = 0.5 * (0.9 * 1.0 * (1.0 / np.log2(3.0)))
    assert np.isclose(div_val, expected_linear - expected_penalty, atol=1e-4)


# -------------------------------------------------------------------------
# 2. Stage 1: Linear Assignment Solver (scipy Hungarian Algorithm)
# -------------------------------------------------------------------------

def test_linear_assignment_solver_exact_optimum():
    cands = [make_candidate(f"doc_{i}", f"Title {i}") for i in range(5)]
    utils = [0.20, 0.95, 0.40, 0.85, 0.60]

    solver = LinearAssignmentSolver()
    ranking, obj_val, status, details = solver.solve(
        candidates=cands,
        utilities=utils,
        k=3,
        objective_type=ObjectiveType.LINEAR_POSITIONAL,
    )

    assert status == SolverStatus.OPTIMAL
    # Top 3 highest utilities should be placed in order: doc 1 (0.95), doc 3 (0.85), doc 4 (0.60)
    assert ranking == [1, 3, 4]
    discounts = calculate_position_discounts(3)
    expected_obj = 0.95 * discounts[0] + 0.85 * discounts[1] + 0.60 * discounts[2]
    assert np.isclose(obj_val, expected_obj, atol=1e-4)
    assert details["algorithm"] == "Hungarian / linear_sum_assignment"


# -------------------------------------------------------------------------
# 3. Stage 2: Pairwise Diversity ILP Solver (scipy MILP)
# -------------------------------------------------------------------------

def test_pairwise_diversity_milp_solver_avoids_redundancy():
    """
    Candidate 0: u = 0.90, concepts: [A]
    Candidate 1: u = 0.86, concepts: [A] (Near duplicate with 0, Sim = 1.0)
    Candidate 2: u = 0.80, concepts: [B] (Diverse from 0, Sim = 0.0)
    Candidate 3: u = 0.20, concepts: [C]

    Under pure linear assignment: picks [0, 1]
    Under Stage 2 MILP with diversity lambda=0.5:
      Pair [0, 1]: obj = 0.90*1.0 + 0.86*0.6309 - 0.5*1.0*1.0*0.6309 = 0.90 + 0.5426 - 0.3155 = 1.1271
      Pair [0, 2]: obj = 0.90*1.0 + 0.80*0.6309 - 0.5*0.0*... = 0.90 + 0.5047 = 1.4047
    MILP MUST pick [0, 2] instead of [0, 1]!
    """
    cands = [
        make_candidate("doc_0", "Paper A1", concepts=["transformers"]),
        make_candidate("doc_1", "Paper A2", concepts=["transformers"]),
        make_candidate("doc_2", "Paper B1", concepts=["diffusion"]),
        make_candidate("doc_3", "Paper C1", concepts=["robotics"]),
    ]
    utils = [0.90, 0.86, 0.80, 0.20]
    sim_mat = compute_candidate_similarity_matrix(cands)
    # Ensure similarity between 0 and 1 is 1.0, and 0 with 2 is 0.0
    sim_mat[0, 1] = 1.0
    sim_mat[1, 0] = 1.0
    sim_mat[0, 2] = 0.0
    sim_mat[2, 0] = 0.0

    milp_solver = PairwiseDiversityILPSolver()
    ranking, obj_val, status, details = milp_solver.solve(
        candidates=cands,
        utilities=utils,
        k=2,
        objective_type=ObjectiveType.PAIRWISE_DIVERSITY,
        similarity_matrix=sim_mat,
        diversity_lambda=0.5,
    )

    assert status == SolverStatus.OPTIMAL
    # Must select 0 and 2 for diversity
    assert ranking == [0, 2]
    # Check that [0, 2] objective exceeds [0, 1]
    discounts = calculate_position_discounts(2)
    obj_01 = evaluate_ranking_objective([0, 1], utils, discounts, ObjectiveType.PAIRWISE_DIVERSITY, sim_mat, 0.5)
    assert obj_val > obj_01


def test_milp_solver_enforces_required_document_constraint():
    """
    Candidate 3 has lower utility, but is marked as required by constraints.
    MILP formulation includes sum_j x_{req, j} == 1 constraint.
    """
    cands = [make_candidate(f"doc_{i}", f"Title {i}") for i in range(4)]
    utils = [0.95, 0.90, 0.85, 0.30]  # doc_3 has low utility
    cons = SearchConstraints(required_doc_ids=["doc_3"])

    milp_solver = PairwiseDiversityILPSolver()
    ranking, obj_val, status, details = milp_solver.solve(
        candidates=cands,
        utilities=utils,
        k=2,
        objective_type=ObjectiveType.LINEAR_POSITIONAL,
        constraints=cons,
    )

    assert status == SolverStatus.OPTIMAL
    # Candidate index 3 MUST be present in the top-2 ranking
    assert 3 in ranking
    # The other candidate should be doc 0 (highest utility)
    assert 0 in ranking


# -------------------------------------------------------------------------
# 4. Exhaustive Permutation Enumeration & Exact Optimum Verification
# -------------------------------------------------------------------------

def test_exhaustive_solver_matches_milp_exact_optimum():
    """
    For small instances (N=5, K=3), compare MILP with Exhaustive search.
    Both must yield the exact same objective value and ranking.
    """
    cands = [make_candidate(f"doc_{i}", f"Paper {i}", concepts=[f"concept_{i % 2}"]) for i in range(5)]
    utils = [0.88, 0.84, 0.79, 0.65, 0.50]
    sim_mat = compute_candidate_similarity_matrix(cands)

    exhaustive = ExhaustivePermutationSolver(max_candidates=10)
    ex_rank, ex_obj, ex_status, _ = exhaustive.solve(
        candidates=cands,
        utilities=utils,
        k=3,
        objective_type=ObjectiveType.PAIRWISE_DIVERSITY,
        similarity_matrix=sim_mat,
        diversity_lambda=0.3,
    )
    assert ex_status == SolverStatus.OPTIMAL

    milp_solver = PairwiseDiversityILPSolver()
    milp_rank, milp_obj, milp_status, _ = milp_solver.solve(
        candidates=cands,
        utilities=utils,
        k=3,
        objective_type=ObjectiveType.PAIRWISE_DIVERSITY,
        similarity_matrix=sim_mat,
        diversity_lambda=0.3,
    )
    assert milp_status == SolverStatus.OPTIMAL

    # Objectives must match exactly down to 1e-4
    assert np.isclose(ex_obj, milp_obj, atol=1e-4)
    assert ex_rank == milp_rank


# -------------------------------------------------------------------------
# 5. Approximation Gap Recording (Greedy vs Exact)
# -------------------------------------------------------------------------

def test_approximation_gap_recorded_for_greedy_solver():
    """
    When an approximate solver runs, OptimizationVerifier must compute:
    Delta U = U(R_exact) - U(R_approx) >= 0.0
    and record verification status APPROXIMATION_GAP_RECORDED.
    """
    cands = [make_candidate(f"doc_{i}", f"Paper {i}", concepts=[f"concept_{i % 2}"]) for i in range(6)]
    utils = [0.90, 0.85, 0.80, 0.70, 0.60, 0.50]
    sim_mat = compute_candidate_similarity_matrix(cands)

    greedy = GreedyDiversitySolver()
    g_rank, g_obj, g_status, g_details = greedy.solve(
        candidates=cands,
        utilities=utils,
        k=3,
        objective_type=ObjectiveType.PAIRWISE_DIVERSITY,
        similarity_matrix=sim_mat,
        diversity_lambda=0.4,
    )

    verifier = OptimizationVerifier(exhaustive_threshold=10)
    report = verifier.verify(
        candidates=cands,
        ranked_indices=g_rank,
        utilities=utils,
        reported_objective=g_obj,
        solver_used=SolverType.GREEDY,
        solver_status=g_status,
        objective_type=ObjectiveType.PAIRWISE_DIVERSITY,
        k=3,
        similarity_matrix=sim_mat,
        diversity_lambda=0.4,
        solver_details=g_details,
    )

    assert report.is_verified is True
    assert report.exhaustive_comparison_performed is True
    assert report.exact_reference_objective is not None
    assert report.approximation_gap is not None
    assert report.approximation_gap >= 0.0
    assert report.verification_status in (
        VerificationStatus.APPROXIMATION_GAP_RECORDED,
        VerificationStatus.VERIFIED_OPTIMAL,
    )


# -------------------------------------------------------------------------
# 6. Constraint Violation Verification
# -------------------------------------------------------------------------

def test_verifier_flags_hard_constraint_violations():
    # Candidate published in 2018
    cands = [make_candidate("doc_old", "Old Paper", year=2018)]
    utils = [0.95]
    cons = SearchConstraints(min_year=2021)

    verifier = OptimizationVerifier()
    report = verifier.verify(
        candidates=cands,
        ranked_indices=[0],
        utilities=utils,
        reported_objective=0.95,
        solver_used=SolverType.LINEAR_ASSIGNMENT,
        solver_status=SolverStatus.OPTIMAL,
        objective_type=ObjectiveType.LINEAR_POSITIONAL,
        k=1,
        constraints=cons,
    )

    assert report.is_verified is False
    assert report.constraints_satisfied is False
    assert report.verification_status == VerificationStatus.CONSTRAINT_VIOLATION
    assert any("min_year" in v for v in report.constraint_violations)


# -------------------------------------------------------------------------
# 7. End-to-End GlobalRankingOptimizer Pipeline
# -------------------------------------------------------------------------

def test_global_ranking_optimizer_end_to_end():
    cands = [
        make_candidate("p1", "Self-Attention in Deep Networks", year=2022, citations=500, code_url="https://github.com/p1"),
        make_candidate("p2", "Attention Variants for Transformers", year=2023, citations=300, code_url="https://github.com/p2"),
        make_candidate("p3", "Diffusion Models Survey", year=2024, citations=150, code_url="https://github.com/p3"),
        make_candidate("p4", "Obsolete NLP Method", year=2015, citations=10, code_url=None),
    ]

    prefs = UserPreferences(importance_relevance=0.8, importance_recency=0.5)
    fb = [FeedbackItem(document_id="p1", feedback_type=FeedbackType.RELEVANT)]
    cons = SearchConstraints(min_year=2020)  # p4 will be inadmissible

    problem = OptimizationProblem(
        candidates=cands,
        k=2,
        objective_type=ObjectiveType.LINEAR_POSITIONAL,
        preferences=prefs,
        feedback_items=fb,
        constraints=cons,
    )

    optimizer = GlobalRankingOptimizer()
    result: OptimizationResult = optimizer.optimize(problem)

    assert len(result.ranked_candidates) == 2
    assert result.solver_status == SolverStatus.OPTIMAL
    assert result.solver_used == SolverType.LINEAR_ASSIGNMENT
    assert result.verification.is_verified is True
    assert result.verification.constraints_satisfied is True
    assert result.verification.candidate_count == 3  # 3 admissible candidates evaluated
    assert result.guarantee_scope == OPTIMALITY_GUARANTEE_STATEMENT

    # Candidate 4 (2015) must NOT appear in ranked list
    ranked_ids = [c.document.internal_id for c in result.ranked_candidates]
    assert "p4" not in ranked_ids


def test_optimizer_raises_on_contradiction():
    cands = [make_candidate("p1", "Paper 1")]
    # Contradiction: min_year > max_year
    cons = SearchConstraints(min_year=2024, max_year=2020)
    problem = OptimizationProblem(
        candidates=cands,
        k=1,
        constraints=cons,
    )
    optimizer = GlobalRankingOptimizer()
    with pytest.raises(ConstraintContradictionError):
        optimizer.optimize(problem)


# -------------------------------------------------------------------------
# 8. REST API Verification (/api/v1/optimize and /api/v1/search)
# -------------------------------------------------------------------------

def test_api_optimize_endpoint():
    cand_data = make_candidate("doc_api", "API Optimization Paper").model_dump()
    payload = {
        "candidates": [cand_data],
        "k": 1,
        "objective_type": "linear_positional",
        "diversity_lambda": 0.2,
    }

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "ranked_candidates" in data
    assert len(data["ranked_candidates"]) == 1
    assert data["solver_used"] == "linear_assignment"
    assert data["verification"]["is_verified"] is True
    assert "globally optimal over the retrieved candidate set" in data["guarantee_scope"]


def test_api_search_with_optimization_flag(monkeypatch):
    """Test that search endpoint honors optimize=True."""
    mock_papers = [
        make_candidate("doc_search_1", "Search Paper 1", concepts=["transformers"]).document,
        make_candidate("doc_search_2", "Search Paper 2", concepts=["graph nn"]).document,
    ]

    async def mock_retrieve(self, query, documents, candidate_limit):
        from app.schemas.retrieval import RetrievalMetadata, RetrievedCandidate
        return [
            RetrievedCandidate(
                document=d,
                retrieval_score=0.9,
                retrieval_rank=i + 1,
                metadata=RetrievalMetadata(
                    bm25_rank=i+1, bm25_score=5.0,
                    sbert_rank=i+1, sbert_score=0.8,
                    rrf_rank=i+1, rrf_score=0.03,
                )
            )
            for i, d in enumerate(mock_papers)
        ]

    from app.retrieval.pipeline import HybridRetrievalPipeline
    monkeypatch.setattr(HybridRetrievalPipeline, "retrieve", mock_retrieve)

    payload = {
        "query": "transformers and graphs",
        "limit": 2,
        "optimize": True,
        "optimization_objective": "pairwise_diversity",
        "diversity_lambda": 0.3,
    }

    response = client.post("/api/v1/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["optimization_result"] is not None
    assert data["optimization_result"]["solver_used"] == "ilp_diversity"
    assert data["optimization_result"]["verification"]["is_verified"] is True
