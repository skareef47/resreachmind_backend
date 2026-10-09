"""
Large-Scale Optimization Benchmarking and Validation Suite for ResearchMind AI.

Evaluates optimizer correctness, runtime, memory usage, scalability,
and optimality behavior across candidate counts N in [5, 10, 20, 50, 100, 250, 500, 1000].

Formal Guarantee Scope:
"Globally optimal with respect to the retrieved candidate set D and the defined utility function U."
Never claim global optimality over the entire scholarly corpus.
"""

import gc
import math
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.feedback.models import FeedbackItem, FeedbackType
from app.optimization.constraints import SearchConstraints
from app.optimization.models import (
    ObjectiveType,
    OptimizationProblem,
    OptimizationResult,
    OptimizationStatus,
    SolverStatus,
    SolverType,
)
from app.optimization.optimizer import GlobalRankingOptimizer
from app.optimization.solvers import (
    ExhaustivePermutationSolver,
    GreedyDiversitySolver,
    LinearAssignmentSolver,
    PairwiseDiversityILPSolver,
    calculate_position_discounts,
    compute_candidate_similarity_matrix,
    evaluate_ranking_objective,
)
from app.optimization.utility import FormalUtilityModel, UtilityWeights
from app.optimization.verification import OptimizationVerifier
from app.preferences.models import UserPreferences
from app.schemas.paper import ResearchDocument
from app.schemas.retrieval import RetrievalMetadata, RetrievedCandidate


# ---------------------------------------------------------------------------
# 1. Deterministic Synthetic Dataset Generator
# ---------------------------------------------------------------------------

CONCEPTS_POOL = [
    "deep learning",
    "transformers",
    "graph neural networks",
    "reinforcement learning",
    "natural language processing",
    "computer vision",
    "optimization",
    "sparse representations",
    "few-shot learning",
    "self-supervised learning",
]

VENUES_POOL = [
    ("NeurIPS", 1.0),
    ("ICML", 1.0),
    ("ICLR", 1.0),
    ("CVPR", 1.0),
    ("ACL", 1.0),
    ("IEEE TPAMI", 0.8),
    ("AAAI", 0.8),
    ("EMNLP", 0.8),
    ("arXiv", 0.6),
    ("Workshop", 0.4),
]


def generate_deterministic_candidates(
    n: int,
    seed: int = 42,
) -> List[RetrievedCandidate]:
    """
    Generates N fully deterministic candidates with all utility features:
    - relevance
    - preference
    - feedback
    - buildability
    - source quality
    - recency
    - penalties
    """
    rng = np.random.default_rng(seed)
    candidates: List[RetrievedCandidate] = []

    for i in range(n):
        doc_id = f"synth_paper_{i:04d}"
        
        # 1. Relevance: beta distributed to simulate realistic reranker score spread
        rel = float(np.clip(rng.beta(2.5, 2.0), 0.1, 0.98))
        
        # 2. Concepts & Venue for preference & source quality
        c_count = int(rng.integers(1, 4))
        concepts = [str(c) for c in rng.choice(CONCEPTS_POOL, size=c_count, replace=False)]
        venue_idx = int(rng.integers(0, len(VENUES_POOL)))
        venue_name, venue_weight = VENUES_POOL[venue_idx]
        
        # Citations log-normal
        citations = int(rng.lognormal(mean=3.5, sigma=1.2))
        
        # Year: between 2017 and 2025
        year = int(rng.integers(2017, 2026))
        
        # Buildability proxy features
        has_code = bool(rng.random() > 0.45)
        code_url = f"https://github.com/synthetic/repo_{i}" if has_code else None
        open_access = bool(rng.random() > 0.35)
        buildability_score = float(np.clip(0.35 + 0.4 * (1 if has_code else 0) + 0.25 * rng.random(), 0.1, 1.0))
        
        doc = ResearchDocument(
            internal_id=doc_id,
            source="synthetic_benchmark",
            source_id=doc_id,
            title=f"Deterministic Study on {' and '.join(concepts[:2])} #{i}",
            abstract=f"We investigate {' and '.join(concepts)} using empirical evaluations. Code available.",
            authors=[f"Researcher {chr(65 + (i % 26))}"],
            publication_year=year,
            citation_count=citations,
            venue=venue_name,
            concepts=concepts,
            code_url=code_url,
            open_access=open_access,
            buildability_score=buildability_score,
        )
        
        meta = RetrievalMetadata(
            bm25_rank=i + 1,
            bm25_score=float(15.0 - (i * 0.01)),
            sbert_rank=i + 1,
            sbert_score=rel,
            rrf_rank=i + 1,
            rrf_score=float(1.0 / (60 + i + 1)),
        )
        
        cand = RetrievedCandidate(
            document=doc,
            retrieval_score=rel,
            retrieval_rank=i + 1,
            metadata=meta,
            features={
                "cross_encoder_score": rel,
                "buildability": buildability_score,
                "relevance": rel,
            },
        )
        candidates.append(cand)

    return candidates


# ---------------------------------------------------------------------------
# 2. Scenario Generators
# ---------------------------------------------------------------------------

def create_scenario_problem(
    scenario_id: str,
    candidates: List[RetrievedCandidate],
    k: int = 10,
    seed: int = 42,
) -> OptimizationProblem:
    """
    Constructs scenario problem configurations:
    A: Pure positional utility
    B: Preferences
    C: Feedback
    D: Hard constraints
    E: Diversity / coupled penalties
    F: Mixed objective
    """
    target_k = min(k, len(candidates))
    
    if scenario_id == "A":
        # Pure positional utility: linear objective, neutral weights
        return OptimizationProblem(
            candidates=candidates,
            k=target_k,
            objective_type=ObjectiveType.LINEAR_POSITIONAL,
            weights=UtilityWeights(
                w1_relevance=1.0,
                w2_preference_match=0.0,
                w3_feedback_match=0.0,
                w4_buildability=0.0,
                w5_source_quality=0.0,
                w6_recency=0.0,
            ),
        )
        
    elif scenario_id == "B":
        # Preferences: active user domain preferences
        prefs = UserPreferences(
            preferred_concepts=["transformers", "graph neural networks"],
            preferred_venues=["NeurIPS", "ICML"],
            prefer_recent=True,
            prefer_reproducible=True,
        )
        return OptimizationProblem(
            candidates=candidates,
            k=target_k,
            objective_type=ObjectiveType.LINEAR_POSITIONAL,
            preferences=prefs,
            weights=UtilityWeights(
                w1_relevance=0.40,
                w2_preference_match=0.40,
                w3_feedback_match=0.0,
                w4_buildability=0.10,
                w5_source_quality=0.05,
                w6_recency=0.05,
            ),
        )
        
    elif scenario_id == "C":
        # Feedback: specific relevant and not-relevant feedback points
        fb_items = [
            FeedbackItem(document_id=candidates[0].document.internal_id, feedback_type=FeedbackType.RELEVANT, weight=1.0),
            FeedbackItem(document_id=candidates[1].document.internal_id, feedback_type=FeedbackType.NOT_RELEVANT, weight=1.0),
        ]
        if len(candidates) > 4:
            fb_items.append(
                FeedbackItem(document_id=candidates[3].document.internal_id, feedback_type=FeedbackType.RELEVANT, weight=1.0)
            )
        return OptimizationProblem(
            candidates=candidates,
            k=target_k,
            objective_type=ObjectiveType.LINEAR_POSITIONAL,
            feedback_items=fb_items,
            weights=UtilityWeights(
                w1_relevance=0.40,
                w2_preference_match=0.0,
                w3_feedback_match=0.40,
                w4_buildability=0.10,
                w5_source_quality=0.05,
                w6_recency=0.05,
            ),
        )
        
    elif scenario_id == "D":
        # Hard constraints: filter candidates
        cons = SearchConstraints(
            min_year=2019,
            require_code=True,
        )
        return OptimizationProblem(
            candidates=candidates,
            k=target_k,
            objective_type=ObjectiveType.LINEAR_POSITIONAL,
            constraints=cons,
        )
        
    elif scenario_id == "E":
        # Diversity / coupled penalties: quadratic pairwise penalization
        return OptimizationProblem(
            candidates=candidates,
            k=target_k,
            objective_type=ObjectiveType.PAIRWISE_DIVERSITY,
            diversity_lambda=0.25,
            solver_type=SolverType.ILP_DIVERSITY if len(candidates) <= 20 else SolverType.GREEDY,
        )
        
    elif scenario_id == "F":
        # Mixed objective: all components active
        prefs = UserPreferences(
            preferred_concepts=["deep learning", "optimization"],
            preferred_venues=["NeurIPS", "ICLR"],
            prefer_reproducible=True,
        )
        fb_items = [
            FeedbackItem(document_id=candidates[0].document.internal_id, feedback_type=FeedbackType.RELEVANT, weight=1.0),
        ]
        cons = SearchConstraints(
            min_year=2018,
            min_citations=5,
        )
        return OptimizationProblem(
            candidates=candidates,
            k=target_k,
            objective_type=ObjectiveType.LINEAR_POSITIONAL,
            preferences=prefs,
            feedback_items=fb_items,
            constraints=cons,
            weights=UtilityWeights(
                w1_relevance=0.35,
                w2_preference_match=0.20,
                w3_feedback_match=0.15,
                w4_buildability=0.15,
                w5_source_quality=0.10,
                w6_recency=0.05,
            ),
        )
        
    else:
        raise ValueError(f"Unknown scenario_id: {scenario_id}")


# ---------------------------------------------------------------------------
# 3. Measurement & Benchmarking Core
# ---------------------------------------------------------------------------

class BenchmarkResult:
    """Stores benchmark metrics for a single test point."""

    def __init__(
        self,
        n: int,
        k: int,
        scenario: str,
        solver_name: str,
        runtime_ms: float,
        memory_mb: float,
        objective_value: float,
        exact_objective: Optional[float],
        optimality_gap: Optional[float],
        solver_status: str,
        optimization_status: str,
        constraint_count: int = 0,
        notes: str = "",
    ) -> None:
        self.n = n
        self.k = k
        self.scenario = scenario
        self.solver_name = solver_name
        self.runtime_ms = runtime_ms
        self.memory_mb = memory_mb
        self.objective_value = objective_value
        self.exact_objective = exact_objective
        self.optimality_gap = optimality_gap
        self.solver_status = solver_status
        self.optimization_status = optimization_status
        self.constraint_count = constraint_count
        self.notes = notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "k": self.k,
            "scenario": self.scenario,
            "solver": self.solver_name,
            "runtime_ms": round(self.runtime_ms, 3),
            "memory_mb": round(self.memory_mb, 4),
            "objective": round(self.objective_value, 6),
            "exact_objective": round(self.exact_objective, 6) if self.exact_objective is not None else None,
            "gap": round(self.optimality_gap, 6) if self.optimality_gap is not None else 0.0,
            "solver_status": self.solver_status,
            "optimization_status": self.optimization_status,
            "constraint_count": self.constraint_count,
            "notes": self.notes,
        }


def run_single_benchmark(
    n: int,
    scenario_id: str,
    k: int = 10,
    seed: int = 42,
    run_exhaustive_if_small: bool = True,
) -> BenchmarkResult:
    """Executes a single benchmark run with memory and high-precision timing telemetry."""
    candidates = generate_deterministic_candidates(n=n, seed=seed)
    target_k = min(k, n)
    problem = create_scenario_problem(scenario_id=scenario_id, candidates=candidates, k=target_k, seed=seed)

    optimizer = GlobalRankingOptimizer(verifier=OptimizationVerifier(exhaustive_threshold=10))

    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()

    result = optimizer.optimize(problem)

    runtime_ms = (time.perf_counter() - start_time) * 1000.0
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_mb = peak_mem / (1024.0 * 1024.0)

    verif = result.verification
    exact_obj = verif.exact_reference_objective if verif else None
    gap = verif.approximation_gap if verif else None

    if n <= 10 and exact_obj is None and run_exhaustive_if_small and len(result.ranked_candidates) > 0:
        ex_solver = ExhaustivePermutationSolver(max_candidates=10)
        u_model = FormalUtilityModel()
        from app.feedback.matcher import FeedbackMatcher
        matcher = FeedbackMatcher({c.document.internal_id: c.document for c in candidates})
        utilities = [
            u_model.evaluate_candidate(
                c,
                problem.preferences or UserPreferences(),
                matcher,
                problem.feedback_items or [],
                problem.constraints or SearchConstraints(),
                problem.weights,
            ).utility_score
            for c in candidates
        ]
        _, exact_val, _, _ = ex_solver.solve(
            candidates=candidates,
            utilities=utilities,
            k=target_k,
            objective_type=problem.objective_type,
            diversity_lambda=problem.diversity_lambda,
            constraints=problem.constraints,
        )
        exact_obj = exact_val
        gap = max(0.0, round(exact_val - result.objective_value, 6))

    opt_status = result.optimization_status.value if hasattr(result, "optimization_status") else "globally_optimized"

    return BenchmarkResult(
        n=n,
        k=target_k,
        scenario=scenario_id,
        solver_name=result.solver_used.value,
        runtime_ms=runtime_ms,
        memory_mb=memory_mb,
        objective_value=result.objective_value,
        exact_objective=exact_obj,
        optimality_gap=gap,
        solver_status=result.solver_status.value,
        optimization_status=opt_status,
        constraint_count=len(verif.constraint_violations) if verif else 0,
        notes=f"Solved via {result.solver_used.value}; verified={verif.is_verified if verif else False}",
    )
