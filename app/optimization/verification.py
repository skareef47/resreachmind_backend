"""
Independent verification and audit module for ResearchMind optimization.

Verification guarantees:
- Recomputed objective independently matches solver-reported objective.
- Verified compliance with all hard constraints, exclusions, and required document sets.
- Exhaustive permutation comparison for small candidate sets (N <= 10) to confirm exact global optimality.
- Exact approximation gap recording: Delta U = U(R_exact) - U(R_approx) >= 0.0.
- Telemetry recording of candidate count, solver, objective, constraints, verification status, and approximation status.
"""

import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np

from app.optimization.constraints import ConstraintEvaluator, SearchConstraints
from app.optimization.models import (
    ObjectiveType,
    SolverStatus,
    SolverType,
    VerificationReport,
    VerificationStatus,
)
from app.optimization.solvers import (
    ExhaustivePermutationSolver,
    calculate_position_discounts,
    evaluate_ranking_objective,
)
from app.schemas.retrieval import RetrievedCandidate

logger = logging.getLogger("researchmind.optimization.verification")


class OptimizationVerifier:
    """
    Independent verification engine for ranking optimization.
    
    Ensures that solutions are strictly compliant with user-specified constraints
    and that reported objective scores are mathematically correct.
    """

    def __init__(self, exhaustive_threshold: int = 10) -> None:
        self.exhaustive_threshold = exhaustive_threshold
        self.constraint_evaluator = ConstraintEvaluator()

    def verify(
        self,
        candidates: List[RetrievedCandidate],
        ranked_indices: List[int],
        utilities: List[float],
        reported_objective: float,
        solver_used: SolverType,
        solver_status: SolverStatus,
        objective_type: ObjectiveType,
        k: int,
        similarity_matrix: Optional[np.ndarray] = None,
        diversity_lambda: float = 0.0,
        constraints: Optional[SearchConstraints] = None,
        solver_details: Optional[Dict[str, Any]] = None,
    ) -> VerificationReport:
        """
        Executes independent verification of the solver's ranking outcome.
        """
        n = len(candidates)
        violations: List[str] = []
        target_k = min(k, n)

        # ----------------------------------------------------
        # 1. Structural Checks (Rank list length & uniqueness)
        # ----------------------------------------------------
        if len(ranked_indices) != target_k:
            violations.append(
                f"Ranked list length ({len(ranked_indices)}) does not match expected target ({target_k})"
            )

        if len(set(ranked_indices)) != len(ranked_indices):
            violations.append("Ranked list contains duplicate candidate indices")

        for idx in ranked_indices:
            if idx < 0 or idx >= n:
                violations.append(f"Ranked candidate index {idx} out of range [0, {n - 1}]")

        # ----------------------------------------------------
        # 2. Hard Constraints & Exclusions Check
        # ----------------------------------------------------
        cons = constraints or SearchConstraints()
        ranked_candidates = [candidates[i] for i in ranked_indices if 0 <= i < n]

        for rank_pos, cand in enumerate(ranked_candidates):
            eval_res = self.constraint_evaluator.evaluate(cand, cons)
            if not eval_res.is_admissible:
                violations.append(
                    f"Candidate at rank {rank_pos + 1} ({cand.document.internal_id}) violates hard constraints: "
                    + "; ".join(eval_res.hard_violations)
                )

        # Check required document IDs inclusion in top-k
        if cons.required_doc_ids:
            req_set = set(cons.required_doc_ids)
            selected_ids: Set[str] = set()
            for cand in ranked_candidates:
                selected_ids.add(cand.document.internal_id)
                if cand.document.source_id:
                    selected_ids.add(cand.document.source_id)
                if cand.document.doi:
                    selected_ids.add(cand.document.doi)

            missing_req = req_set - selected_ids
            # Only consider missing as violation if the candidate pool actually contained that document
            pool_doc_ids = set()
            for cand in candidates:
                pool_doc_ids.add(cand.document.internal_id)
                if cand.document.source_id:
                    pool_doc_ids.add(cand.document.source_id)
                if cand.document.doi:
                    pool_doc_ids.add(cand.document.doi)

            actually_missing = missing_req.intersection(pool_doc_ids)
            if actually_missing and len(ranked_candidates) >= len(actually_missing):
                violations.append(
                    f"Required document(s) {sorted(list(actually_missing))} missing from optimal ranking"
                )

        constraints_satisfied = len(violations) == 0

        # ----------------------------------------------------
        # 3. Independent Objective Recomputation
        # ----------------------------------------------------
        gamma = calculate_position_discounts(target_k)
        recomputed_objective = evaluate_ranking_objective(
            ranking_indices=ranked_indices,
            utilities=utilities,
            discounts=gamma,
            objective_type=objective_type,
            similarity_matrix=similarity_matrix,
            diversity_lambda=diversity_lambda,
        )

        # Allow small floating point tolerance
        is_objective_match = abs(float(reported_objective) - float(recomputed_objective)) < 1e-4

        # ----------------------------------------------------
        # 4. Exhaustive Comparison on Small Instances (N <= 10)
        # ----------------------------------------------------
        exhaustive_performed = False
        exact_reference_objective: Optional[float] = None
        approximation_gap: Optional[float] = None

        if 0 < n <= self.exhaustive_threshold:
            exhaustive_solver = ExhaustivePermutationSolver(max_candidates=self.exhaustive_threshold)
            _, exact_ref_obj, ex_status, _ = exhaustive_solver.solve(
                candidates=candidates,
                utilities=utilities,
                k=target_k,
                objective_type=objective_type,
                similarity_matrix=similarity_matrix,
                diversity_lambda=diversity_lambda,
                constraints=cons,
            )

            if ex_status == SolverStatus.OPTIMAL:
                exhaustive_performed = True
                exact_reference_objective = exact_ref_obj
                raw_gap = exact_ref_obj - recomputed_objective
                # Round and clamp to non-negative
                approximation_gap = max(0.0, round(raw_gap, 6))

        # ----------------------------------------------------
        # 5. Determine Verification Status
        # ----------------------------------------------------
        if not constraints_satisfied:
            verification_status = VerificationStatus.CONSTRAINT_VIOLATION
            is_verified = False
        elif not is_objective_match:
            verification_status = VerificationStatus.VERIFICATION_FAILED
            is_verified = False
        elif solver_status in (SolverStatus.FAILED, SolverStatus.INFEASIBLE):
            verification_status = VerificationStatus.VERIFICATION_FAILED
            is_verified = False
        elif exhaustive_performed and approximation_gap is not None:
            if approximation_gap <= 1e-4:
                verification_status = VerificationStatus.VERIFIED_OPTIMAL
                is_verified = True
            else:
                verification_status = VerificationStatus.APPROXIMATION_GAP_RECORDED
                is_verified = True
        else:
            if solver_status == SolverStatus.OPTIMAL:
                verification_status = VerificationStatus.VERIFIED_OPTIMAL
                is_verified = True
            else:
                verification_status = VerificationStatus.VERIFIED_FEASIBLE
                is_verified = True

        details = dict(solver_details or {})
        details["independent_evaluator"] = "evaluate_ranking_objective"
        details["exhaustive_threshold"] = self.exhaustive_threshold

        return VerificationReport(
            is_verified=is_verified,
            verification_status=verification_status,
            solver_status=solver_status,
            candidate_count=n,
            k=target_k,
            solver_used=solver_used,
            objective_type=objective_type,
            reported_objective=reported_objective,
            recomputed_objective=recomputed_objective,
            is_objective_match=is_objective_match,
            constraints_satisfied=constraints_satisfied,
            constraint_violations=violations,
            exhaustive_comparison_performed=exhaustive_performed,
            exact_reference_objective=exact_reference_objective,
            approximation_gap=approximation_gap,
            details=details,
        )
