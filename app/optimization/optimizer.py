"""
Global Ranking Optimizer for ResearchMind AI.

Solves the core ranking optimization problem:
R* = argmax_{R in S_n} U(R | q, D, F, P, C)

Formal Guarantee Scope:
"globally optimal over the retrieved candidate set under the explicitly defined utility and constraints"
Do not claim universal optimality over all possible documents or undefined human preferences.

Architecture:
Stage 1: LinearAssignmentSolver (Hungarian algorithm via scipy.optimize.linear_sum_assignment)
Stage 2: PairwiseDiversityILPSolver (MILP via scipy.optimize.milp with McCormick linearization)
Ground Truth: ExhaustivePermutationSolver for exact verification on small candidate sets
Independent Verification: OptimizationVerifier ensuring constraint satisfaction and calculating Delta U
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.feedback.matcher import FeedbackMatcher
from app.feedback.validator import validate_feedback
from app.optimization.constraints import (
    ConstraintEvaluator,
    SearchConstraints,
    validate_constraints,
)
from app.optimization.models import (
    ObjectiveType,
    OptimizationProblem,
    OptimizationResult,
    SolverStatus,
    SolverType,
)
from app.optimization.solvers import (
    BaseRankingSolver,
    ExhaustivePermutationSolver,
    GreedyDiversitySolver,
    LinearAssignmentSolver,
    PairwiseDiversityILPSolver,
    compute_candidate_similarity_matrix,
)
from app.optimization.utility import FormalUtilityModel, UtilityWeights
from app.preferences.models import UserPreferences
from app.schemas.retrieval import RetrievedCandidate
from app.optimization.verification import OptimizationVerifier

logger = logging.getLogger("researchmind.optimization.optimizer")

OPTIMALITY_GUARANTEE_STATEMENT = (
    "globally optimal over the retrieved candidate set under the explicitly defined utility and constraints"
)


class GlobalRankingOptimizer:
    """
    Core ResearchMind Global Ranking Optimizer.
    
    Dispatches to Stage 1 (linear assignment) or Stage 2 (pairwise ILP),
    followed by mandatory independent verification.
    """

    def __init__(
        self,
        utility_model: Optional[FormalUtilityModel] = None,
        verifier: Optional[OptimizationVerifier] = None,
    ) -> None:
        self.utility_model = utility_model or FormalUtilityModel()
        self.verifier = verifier or OptimizationVerifier(exhaustive_threshold=10)
        self.constraint_evaluator = ConstraintEvaluator()

    def _select_solver(
        self,
        problem: OptimizationProblem,
        admissible_count: int,
    ) -> BaseRankingSolver:
        """
        Selects the mathematically appropriate solver.
        
        Rules:
        - If problem explicitly requested a solver, respect it.
        - Stage 1: If objective is LINEAR_POSITIONAL and no complex required-doc constraints,
          use LinearAssignmentSolver (scipy.optimize.linear_sum_assignment).
        - Stage 2: If objective is PAIRWISE_DIVERSITY or constraints require specific docs,
          use PairwiseDiversityILPSolver (scipy.optimize.milp).
        """
        # Explicit override
        req_solver = getattr(problem, "solver_type", None)
        if req_solver == SolverType.EXHAUSTIVE:
            return ExhaustivePermutationSolver(max_candidates=10)
        if req_solver == SolverType.GREEDY:
            return GreedyDiversitySolver()
        if req_solver == SolverType.LINEAR_ASSIGNMENT:
            return LinearAssignmentSolver()
        if req_solver == SolverType.ILP_DIVERSITY:
            return PairwiseDiversityILPSolver()

        # Automatic Stage Dispatch
        has_required_doc_constraint = bool(
            problem.constraints and problem.constraints.required_doc_ids
        )

        if problem.objective_type == ObjectiveType.LINEAR_POSITIONAL and not has_required_doc_constraint:
            # Stage 1: Pure linear assignment
            return LinearAssignmentSolver()
        else:
            # Stage 2: Pairwise diversity or richer ILP constraints
            return PairwiseDiversityILPSolver()

    def optimize(self, problem: OptimizationProblem) -> OptimizationResult:
        """
        Executes formal global optimization and verification.
        
        Steps:
        1. Validates absence of feedback and constraint contradictions.
        2. Evaluates individual utility scores and checks candidate admissibility.
        3. Filters inadmissible candidates (satisfying hard constraints).
        4. Computes candidate similarity matrix.
        5. Solves ranking problem using Stage 1 Hungarian or Stage 2 MILP.
        6. Runs independent verification audit (recomputes objective, checks constraints,
           compares to exhaustive baseline, records approximation gap).
        7. Returns verified OptimizationResult with explicit epistemic boundary.
        """
        prefs = problem.preferences or UserPreferences()
        fb_items = problem.feedback_items or []
        cons = problem.constraints or SearchConstraints()
        weights = problem.weights or self.utility_model.weights

        # 1. Contradiction Validation (Strict: Never silently resolve)
        validate_feedback(fb_items, raise_on_error=True)
        validate_constraints(cons, feedback_items=fb_items, raise_on_error=True)

        all_candidates = problem.candidates
        if not all_candidates:
            # Empty candidate set
            from app.optimization.models import VerificationReport, VerificationStatus

            empty_report = VerificationReport(
                is_verified=True,
                verification_status=VerificationStatus.VERIFIED_OPTIMAL,
                solver_status=SolverStatus.OPTIMAL,
                candidate_count=0,
                k=0,
                solver_used=SolverType.LINEAR_ASSIGNMENT,
                objective_type=problem.objective_type,
                reported_objective=0.0,
                recomputed_objective=0.0,
                is_objective_match=True,
                constraints_satisfied=True,
                constraint_violations=[],
                exhaustive_comparison_performed=False,
                exact_reference_objective=0.0,
                approximation_gap=0.0,
                details={"message": "Empty candidate pool"},
            )
            return OptimizationResult(
                ranked_candidates=[],
                objective_value=0.0,
                solver_used=SolverType.LINEAR_ASSIGNMENT,
                solver_status=SolverStatus.OPTIMAL,
                verification=empty_report,
                guarantee_scope=OPTIMALITY_GUARANTEE_STATEMENT,
            )

        # 2. Build FeedbackMatcher
        reference_corpus = {c.document.internal_id: c.document for c in all_candidates}
        for c in all_candidates:
            if c.document.source_id:
                reference_corpus[c.document.source_id] = c.document
            if c.document.doi:
                reference_corpus[c.document.doi] = c.document
        fb_matcher = FeedbackMatcher(reference_corpus=reference_corpus)

        # 3. Evaluate utilities & identify admissible candidates
        admissible_candidates: List[RetrievedCandidate] = []
        admissible_utilities: List[float] = []

        for cand in all_candidates:
            scores = self.utility_model.evaluate_candidate(
                candidate=cand,
                preferences=prefs,
                feedback_matcher=fb_matcher,
                feedback_items=fb_items,
                constraints=cons,
                weights=weights,
            )

            if scores.is_admissible:
                cand_copy = cand.model_copy(deep=True)
                if cand_copy.features is None:
                    cand_copy.features = {}
                cand_copy.features["relevance"] = scores.relevance
                cand_copy.features["utility_score"] = scores.utility_score
                cand_copy.features["preference_match"] = scores.preference_match
                cand_copy.features["feedback_match"] = scores.feedback_match
                cand_copy.features["buildability"] = scores.buildability
                cand_copy.features["source_quality"] = scores.source_quality
                cand_copy.features["recency"] = scores.recency
                cand_copy.features["penalties"] = scores.penalties

                # Populate document attributes
                cand_copy.document.paper_id = cand_copy.document.internal_id
                cand_copy.document.relevance_score = scores.relevance
                cand_copy.document.utility_score = scores.utility_score
                cand_copy.document.preference_contribution = round(weights.w2_preference_match * scores.preference_match, 4)
                cand_copy.document.feedback_contribution = round(weights.w3_feedback_match * scores.feedback_match, 4)
                cand_copy.document.constraint_penalty = scores.penalties
                cand_copy.document.buildability_evidence = cand_copy.document.buildability_breakdown

                admissible_candidates.append(cand_copy)
                admissible_utilities.append(scores.utility_score)

        if not admissible_candidates:
            # All candidates violated hard constraints
            from app.optimization.models import VerificationReport, VerificationStatus

            infeasible_report = VerificationReport(
                is_verified=False,
                verification_status=VerificationStatus.CONSTRAINT_VIOLATION,
                solver_status=SolverStatus.INFEASIBLE,
                candidate_count=len(all_candidates),
                k=problem.k,
                solver_used=SolverType.LINEAR_ASSIGNMENT,
                objective_type=problem.objective_type,
                reported_objective=0.0,
                recomputed_objective=0.0,
                is_objective_match=True,
                constraints_satisfied=False,
                constraint_violations=["All candidate documents violated hard constraints or exclusions"],
                exhaustive_comparison_performed=False,
                exact_reference_objective=None,
                approximation_gap=None,
                details={"admissible_candidates": 0},
            )
            return OptimizationResult(
                ranked_candidates=[],
                objective_value=0.0,
                solver_used=SolverType.LINEAR_ASSIGNMENT,
                solver_status=SolverStatus.INFEASIBLE,
                verification=infeasible_report,
                guarantee_scope=OPTIMALITY_GUARANTEE_STATEMENT,
            )

        # 4. Pairwise similarity matrix
        sim_matrix = compute_candidate_similarity_matrix(admissible_candidates)

        # 5. Solver Selection and Execution
        solver = self._select_solver(problem, len(admissible_candidates))
        ranked_indices, reported_obj, solver_status, details = solver.solve(
            candidates=admissible_candidates,
            utilities=admissible_utilities,
            k=problem.k,
            objective_type=problem.objective_type,
            similarity_matrix=sim_matrix,
            diversity_lambda=problem.diversity_lambda,
            constraints=cons,
        )

        # Build ranked candidate output list
        ranked_candidates: List[RetrievedCandidate] = []
        for rank_pos, cand_idx in enumerate(ranked_indices):
            c = admissible_candidates[cand_idx].model_copy(deep=True)
            if c.features is None:
                c.features = {}
            c.features["optimal_rank"] = rank_pos + 1
            c.document.rank = rank_pos + 1
            ranked_candidates.append(c)

        # 6. Independent Verification Audit
        verification_report = self.verifier.verify(
            candidates=admissible_candidates,
            ranked_indices=ranked_indices,
            utilities=admissible_utilities,
            reported_objective=reported_obj,
            solver_used=solver.solver_type,
            solver_status=solver_status,
            objective_type=problem.objective_type,
            k=problem.k,
            similarity_matrix=sim_matrix,
            diversity_lambda=problem.diversity_lambda,
            constraints=cons,
            solver_details=details,
        )

        return OptimizationResult(
            ranked_candidates=ranked_candidates,
            objective_value=reported_obj,
            solver_used=solver.solver_type,
            solver_status=solver_status,
            optimization_status=verification_report.optimization_status,
            verification=verification_report,
            guarantee_scope=OPTIMALITY_GUARANTEE_STATEMENT,
        )
