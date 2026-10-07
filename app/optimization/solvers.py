"""
Optimization solvers for ResearchMind ranking.

Stage 1:
Linear assignment solver using scipy.optimize.linear_sum_assignment (Hungarian algorithm)
for position-discounted linear utility objectives.

Stage 2:
Pairwise diversity / interaction solver using Mixed-Integer Linear Programming (MILP)
with scipy.optimize.milp and McCormick envelope linearization.

Exhaustive solver:
Permutation enumeration for small candidate pools (N <= 10) providing exact
provable ground-truth for verification.

Greedy solver:
Submodular marginal gain approximation baseline for large candidate pools.
"""

from abc import ABC, abstractmethod
import itertools
import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp

from app.optimization.constraints import SearchConstraints
from app.optimization.models import ObjectiveType, SolverStatus, SolverType
from app.schemas.retrieval import RetrievedCandidate

logger = logging.getLogger("researchmind.optimization.solvers")


def calculate_position_discounts(k: int) -> np.ndarray:
    """
    Computes standard Discounted Cumulative Gain (DCG) position discounts.
    gamma_j = 1.0 / log2(2 + j) for j in [0, k-1].
    
    Rank 1 (j=0): gamma_0 = 1.0
    Rank 2 (j=1): gamma_1 = 1.0 / log2(3) ~= 0.6309
    Rank 3 (j=2): gamma_2 = 1.0 / log2(4) = 0.5
    """
    if k <= 0:
        return np.array([], dtype=np.float64)
    return np.array([1.0 / np.log2(2.0 + float(j)) for j in range(k)], dtype=np.float64)


def compute_candidate_similarity_matrix(candidates: List[RetrievedCandidate]) -> np.ndarray:
    """
    Computes an N x N pairwise similarity matrix S in [0.0, 1.0].
    
    Priority:
    1. SBERT dense embedding cosine similarity (if embeddings present in features)
    2. Concept Jaccard similarity
    3. Title token Jaccard similarity fallback
    """
    n = len(candidates)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    sim = np.eye(n, dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            doc_i = candidates[i].document
            doc_j = candidates[j].document
            s_val = 0.0

            # 1. Concept overlap Jaccard
            if doc_i.concepts and doc_j.concepts:
                c_i = {c.lower().strip() for c in doc_i.concepts}
                c_j = {c.lower().strip() for c in doc_j.concepts}
                union = c_i.union(c_j)
                if union:
                    s_val = len(c_i.intersection(c_j)) / len(union)

            # 2. Title token Jaccard fallback/blend
            t_i = set(doc_i.title.lower().split())
            t_j = set(doc_j.title.lower().split())
            t_union = t_i.union(t_j)
            t_jaccard = (len(t_i.intersection(t_j)) / len(t_union)) if t_union else 0.0

            s_val = max(s_val, t_jaccard)
            s_val = float(np.clip(s_val, 0.0, 1.0))

            sim[i, j] = s_val
            sim[j, i] = s_val

    return sim


def evaluate_ranking_objective(
    ranking_indices: List[int],
    utilities: List[float],
    discounts: np.ndarray,
    objective_type: ObjectiveType = ObjectiveType.LINEAR_POSITIONAL,
    similarity_matrix: Optional[np.ndarray] = None,
    diversity_lambda: float = 0.0,
) -> float:
    """
    Independently evaluates objective function value for a given ranking permutation.
    
    Linear:
      U(R) = sum_{j=0}^{K-1} u_{pi_j} * gamma_j
      
    Pairwise Diversity:
      U(R) = sum_{j=0}^{K-1} u_{pi_j} * gamma_j - lambda * sum_{0 <= j < l < K} S_{pi_j, pi_l} * gamma_j * gamma_l
    """
    k = len(ranking_indices)
    if k == 0 or len(utilities) == 0:
        return 0.0

    gamma = discounts[:k]
    # Linear utility component
    linear_val = sum(utilities[idx] * gamma[j] for j, idx in enumerate(ranking_indices))

    if objective_type != ObjectiveType.PAIRWISE_DIVERSITY or diversity_lambda <= 0.0 or similarity_matrix is None:
        return float(round(linear_val, 6))

    # Pairwise redundancy penalty
    penalty = 0.0
    for j in range(k):
        for l in range(j + 1, k):
            idx_j = ranking_indices[j]
            idx_l = ranking_indices[l]
            penalty += similarity_matrix[idx_j, idx_l] * gamma[j] * gamma[l]

    total = linear_val - (diversity_lambda * penalty)
    return float(round(total, 6))


class BaseRankingSolver(ABC):
    """Abstract interface for ranking optimization solvers."""

    solver_type: SolverType

    @abstractmethod
    def solve(
        self,
        candidates: List[RetrievedCandidate],
        utilities: List[float],
        k: int,
        objective_type: ObjectiveType = ObjectiveType.LINEAR_POSITIONAL,
        similarity_matrix: Optional[np.ndarray] = None,
        diversity_lambda: float = 0.2,
        constraints: Optional[SearchConstraints] = None,
    ) -> Tuple[List[int], float, SolverStatus, Dict[str, Any]]:
        """
        Solve optimal candidate ranking.
        
        Returns:
            - ranking_indices: List of candidate indices ordered from rank 1 to K
            - objective_value: Evaluated objective value U(R)
            - status: Solver execution status (OPTIMAL, APPROXIMATE, INFEASIBLE, FAILED)
            - details: Diagnostics and metadata
        """
        pass


class LinearAssignmentSolver(BaseRankingSolver):
    """
    Stage 1: Linear Assignment Solver using scipy.optimize.linear_sum_assignment.
    
    Formulation:
    Maximize sum_{i=0}^{N-1} sum_{j=0}^{K-1} u_i * gamma_j * x_{i, j}
    Cost matrix C_{i, j} = - (u_i * gamma_j)
    
    Guarantees global polynomial-time exact optimality for linear positional objectives.
    """

    solver_type = SolverType.LINEAR_ASSIGNMENT

    def solve(
        self,
        candidates: List[RetrievedCandidate],
        utilities: List[float],
        k: int,
        objective_type: ObjectiveType = ObjectiveType.LINEAR_POSITIONAL,
        similarity_matrix: Optional[np.ndarray] = None,
        diversity_lambda: float = 0.0,
        constraints: Optional[SearchConstraints] = None,
    ) -> Tuple[List[int], float, SolverStatus, Dict[str, Any]]:
        n = len(candidates)
        if n == 0 or k <= 0:
            return [], 0.0, SolverStatus.OPTIMAL, {"message": "Empty candidate pool or k=0"}

        target_k = min(k, n)
        gamma = calculate_position_discounts(target_k)

        # Cost matrix: shape (N, target_k). linear_sum_assignment minimizes cost.
        cost_matrix = np.zeros((n, target_k), dtype=np.float64)
        for i in range(n):
            for j in range(target_k):
                cost_matrix[i, j] = -(utilities[i] * gamma[j])

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Sort assignments by rank position (col_ind from 0 to target_k - 1)
        rank_pairs = sorted(zip(col_ind, row_ind), key=lambda x: x[0])
        ranking = [int(cand_idx) for _, cand_idx in rank_pairs]

        sim_mat = similarity_matrix if similarity_matrix is not None else compute_candidate_similarity_matrix(candidates)
        obj_val = evaluate_ranking_objective(
            ranking_indices=ranking,
            utilities=utilities,
            discounts=gamma,
            objective_type=objective_type,
            similarity_matrix=sim_mat,
            diversity_lambda=diversity_lambda,
        )

        details = {
            "algorithm": "Hungarian / linear_sum_assignment",
            "candidate_count": n,
            "target_k": target_k,
            "is_exact": objective_type == ObjectiveType.LINEAR_POSITIONAL,
        }
        return ranking, obj_val, SolverStatus.OPTIMAL, details


class PairwiseDiversityILPSolver(BaseRankingSolver):
    """
    Stage 2: Pairwise Interactions and Diversity Solver using Mixed-Integer Linear Programming (MILP).
    
    Formulation:
    Variables:
      x_{i, j} in {0, 1}: Candidate i placed at rank position j.
      z_{i, m, j, l} in {0, 1}: Candidate i at position j AND candidate m at position l (for i != m, j < l).
      
    Objective:
      Minimize c^T v where:
        c(x_{i, j}) = - (u_i * gamma_j)
        c(z_{i, m, j, l}) = lambda * S_{i, m} * gamma_j * gamma_l
        
    Linearization constraints (McCormick envelope):
      z_{i, m, j, l} >= x_{i, j} + x_{m, l} - 1  =>  z_{i, m, j, l} - x_{i, j} - x_{m, l} >= -1
      
    Structural constraints:
      sum_{i=0}^{N-1} x_{i, j} == 1 for all j in [0, K-1] (each position has exactly 1 candidate)
      sum_{j=0}^{K-1} x_{i, j} <= 1 for all i in [0, N-1] (each candidate selected at most once)
      
    Required constraints:
      For required document IDs: sum_{j=0}^{K-1} x_{req, j} == 1
    """

    solver_type = SolverType.ILP_DIVERSITY

    def __init__(self, time_limit_sec: float = 10.0) -> None:
        self.time_limit_sec = time_limit_sec

    def solve(
        self,
        candidates: List[RetrievedCandidate],
        utilities: List[float],
        k: int,
        objective_type: ObjectiveType = ObjectiveType.PAIRWISE_DIVERSITY,
        similarity_matrix: Optional[np.ndarray] = None,
        diversity_lambda: float = 0.2,
        constraints: Optional[SearchConstraints] = None,
    ) -> Tuple[List[int], float, SolverStatus, Dict[str, Any]]:
        n = len(candidates)
        if n == 0 or k <= 0:
            return [], 0.0, SolverStatus.OPTIMAL, {"message": "Empty candidate pool"}

        target_k = min(k, n)
        gamma = calculate_position_discounts(target_k)

        sim_mat = similarity_matrix
        if sim_mat is None:
            sim_mat = compute_candidate_similarity_matrix(candidates)

        # Candidate pairs (i, m) with i != m and position pairs (j, l) with j < l
        cand_pairs = [(i, m) for i in range(n) for m in range(n) if i != m]
        pos_pairs = [(j, l) for j in range(target_k) for l in range(j + 1, target_k)]

        n_x = n * target_k
        n_z = len(cand_pairs) * len(pos_pairs)
        n_vars = n_x + n_z

        c = np.zeros(n_vars, dtype=np.float64)

        def x_idx(i_cand: int, j_pos: int) -> int:
            return i_cand * target_k + j_pos

        # Linear objective coefficients (minimizing -utility)
        for i in range(n):
            for j in range(target_k):
                c[x_idx(i, j)] = -(utilities[i] * gamma[j])

        # Quadratic penalty coefficients via linearized z
        z_map: Dict[Tuple[int, int, int, int], int] = {}
        for p_idx, ((i, m), (j, l)) in enumerate([(cp, pp) for cp in cand_pairs for pp in pos_pairs]):
            var_idx = n_x + p_idx
            z_map[(i, m, j, l)] = var_idx
            c[var_idx] = float(diversity_lambda * sim_mat[i, m] * gamma[j] * gamma[l])

        A_rows: List[np.ndarray] = []
        b_l: List[float] = []
        b_u: List[float] = []

        # 1. Exactly one candidate per position: sum_i x_{i, j} == 1
        for j in range(target_k):
            row = np.zeros(n_vars, dtype=np.float64)
            for i in range(n):
                row[x_idx(i, j)] = 1.0
            A_rows.append(row)
            b_l.append(1.0)
            b_u.append(1.0)

        # 2. Each candidate at most once: sum_j x_{i, j} <= 1
        for i in range(n):
            row = np.zeros(n_vars, dtype=np.float64)
            for j in range(target_k):
                row[x_idx(i, j)] = 1.0
            A_rows.append(row)
            b_l.append(0.0)
            b_u.append(1.0)

        # 3. Required document constraint: sum_j x_{req, j} == 1
        if constraints and constraints.required_doc_ids:
            req_set = set(constraints.required_doc_ids)
            for i, cand in enumerate(candidates):
                doc_ids = {cand.document.internal_id, cand.document.source_id}
                if cand.document.doi:
                    doc_ids.add(cand.document.doi)
                if doc_ids.intersection(req_set):
                    row = np.zeros(n_vars, dtype=np.float64)
                    for j in range(target_k):
                        row[x_idx(i, j)] = 1.0
                    A_rows.append(row)
                    b_l.append(1.0)
                    b_u.append(1.0)

        # 4. Linearization of z_{i, m, j, l} >= x_{i, j} + x_{m, l} - 1
        # => z - x_{i, j} - x_{m, l} >= -1
        for (i, m), (j, l) in [(cp, pp) for cp in cand_pairs for pp in pos_pairs]:
            row = np.zeros(n_vars, dtype=np.float64)
            var_z = z_map[(i, m, j, l)]
            row[var_z] = 1.0
            row[x_idx(i, j)] = -1.0
            row[x_idx(m, l)] = -1.0
            A_rows.append(row)
            b_l.append(-1.0)
            b_u.append(np.inf)

        A = np.array(A_rows, dtype=np.float64)
        milp_constraints = LinearConstraint(A, b_l, b_u)
        integrality = np.ones(n_vars, dtype=np.int32)  # Binary variables
        bounds = Bounds(0.0, 1.0)

        options = {"time_limit": self.time_limit_sec}
        try:
            res = milp(c=c, integrality=integrality, bounds=bounds, constraints=milp_constraints, options=options)
            if res.success and res.x is not None:
                x_sol = res.x[:n_x].reshape(n, target_k)
                ranking = [int(np.argmax(x_sol[:, j])) for j in range(target_k)]
                obj_val = evaluate_ranking_objective(
                    ranking_indices=ranking,
                    utilities=utilities,
                    discounts=gamma,
                    objective_type=objective_type,
                    similarity_matrix=sim_mat,
                    diversity_lambda=diversity_lambda,
                )
                details = {
                    "algorithm": "MILP (HiGHS)",
                    "status_code": res.status,
                    "message": res.message,
                    "num_variables": n_vars,
                    "num_constraints": len(b_l),
                }
                return ranking, obj_val, SolverStatus.OPTIMAL, details
            else:
                logger.warning("MILP failed or infeasible: %s, falling back to greedy", res.message)
                status = SolverStatus.INFEASIBLE if "infeasible" in res.message.lower() else SolverStatus.FAILED
                # Fallback to greedy for robustness
                greedy = GreedyDiversitySolver()
                g_rank, g_obj, _, g_details = greedy.solve(
                    candidates=candidates,
                    utilities=utilities,
                    k=k,
                    objective_type=objective_type,
                    similarity_matrix=sim_mat,
                    diversity_lambda=diversity_lambda,
                    constraints=constraints,
                )
                g_details["milp_fallback_reason"] = res.message
                return g_rank, g_obj, status, g_details
        except Exception as exc:
            logger.error("Exception in MILP solver: %s", exc)
            greedy = GreedyDiversitySolver()
            g_rank, g_obj, _, g_details = greedy.solve(
                candidates=candidates,
                utilities=utilities,
                k=k,
                objective_type=objective_type,
                similarity_matrix=sim_mat,
                diversity_lambda=diversity_lambda,
                constraints=constraints,
            )
            g_details["milp_error"] = str(exc)
            return g_rank, g_obj, SolverStatus.FAILED, g_details


class ExhaustivePermutationSolver(BaseRankingSolver):
    """
    Exhaustive Permutation Solver.
    
    Evaluates ALL permutations of candidate indices of length K.
    Guarantees true global optimum for small candidate sets (N <= 10).
    Used as ground truth reference for formal verification and computing
    exact approximation gaps Delta U = U(R_exact) - U(R_approx).
    """

    solver_type = SolverType.EXHAUSTIVE

    def __init__(self, max_candidates: int = 10) -> None:
        self.max_candidates = max_candidates

    def solve(
        self,
        candidates: List[RetrievedCandidate],
        utilities: List[float],
        k: int,
        objective_type: ObjectiveType = ObjectiveType.LINEAR_POSITIONAL,
        similarity_matrix: Optional[np.ndarray] = None,
        diversity_lambda: float = 0.2,
        constraints: Optional[SearchConstraints] = None,
    ) -> Tuple[List[int], float, SolverStatus, Dict[str, Any]]:
        n = len(candidates)
        if n == 0 or k <= 0:
            return [], 0.0, SolverStatus.OPTIMAL, {"message": "Empty candidate pool"}

        target_k = min(k, n)
        gamma = calculate_position_discounts(target_k)

        sim_mat = similarity_matrix
        if sim_mat is None:
            sim_mat = compute_candidate_similarity_matrix(candidates)

        # Identify required candidates if any
        required_indices: Set[int] = set()
        if constraints and constraints.required_doc_ids:
            req_set = set(constraints.required_doc_ids)
            for i, cand in enumerate(candidates):
                doc_ids = {cand.document.internal_id, cand.document.source_id}
                if cand.document.doi:
                    doc_ids.add(cand.document.doi)
                if doc_ids.intersection(req_set):
                    required_indices.add(i)

        best_ranking: Optional[List[int]] = None
        best_obj = -float("inf")
        evaluated_permutations = 0

        # Enumerate all permutations of length target_k
        for perm in itertools.permutations(range(n), target_k):
            # Check required indices constraint
            if required_indices and not required_indices.issubset(set(perm)):
                continue

            evaluated_permutations += 1
            obj = evaluate_ranking_objective(
                ranking_indices=list(perm),
                utilities=utilities,
                discounts=gamma,
                objective_type=objective_type,
                similarity_matrix=sim_mat,
                diversity_lambda=diversity_lambda,
            )
            if obj > best_obj:
                best_obj = obj
                best_ranking = list(perm)

        if best_ranking is None:
            # If no permutation satisfied required_indices, return empty or unconstrained best
            return [], 0.0, SolverStatus.INFEASIBLE, {"message": "No permutation satisfied constraints"}

        details = {
            "algorithm": "Exhaustive Permutation Search",
            "candidate_count": n,
            "target_k": target_k,
            "evaluated_permutations": evaluated_permutations,
            "exact_global_optimum": True,
        }
        return best_ranking, float(round(best_obj, 6)), SolverStatus.OPTIMAL, details


class GreedyDiversitySolver(BaseRankingSolver):
    """
    Greedy Submodular Ranking Solver (Approximation Baseline).
    
    Greedily selects the next candidate maximizing marginal objective gain:
    Delta U = u_i * gamma_j - lambda * sum_{l < j} S_{i, pi_l} * gamma_j * gamma_l
    
    Provides bounded polynomial-time approximation for large candidate pools.
    """

    solver_type = SolverType.GREEDY

    def solve(
        self,
        candidates: List[RetrievedCandidate],
        utilities: List[float],
        k: int,
        objective_type: ObjectiveType = ObjectiveType.PAIRWISE_DIVERSITY,
        similarity_matrix: Optional[np.ndarray] = None,
        diversity_lambda: float = 0.2,
        constraints: Optional[SearchConstraints] = None,
    ) -> Tuple[List[int], float, SolverStatus, Dict[str, Any]]:
        n = len(candidates)
        if n == 0 or k <= 0:
            return [], 0.0, SolverStatus.OPTIMAL, {"message": "Empty candidate pool"}

        target_k = min(k, n)
        gamma = calculate_position_discounts(target_k)

        sim_mat = similarity_matrix
        if sim_mat is None:
            sim_mat = compute_candidate_similarity_matrix(candidates)

        selected: List[int] = []
        remaining = set(range(n))

        # First pick required items if any
        if constraints and constraints.required_doc_ids:
            req_set = set(constraints.required_doc_ids)
            for i in list(remaining):
                cand = candidates[i]
                doc_ids = {cand.document.internal_id, cand.document.source_id}
                if cand.document.doi:
                    doc_ids.add(cand.document.doi)
                if doc_ids.intersection(req_set) and len(selected) < target_k:
                    selected.append(i)
                    remaining.remove(i)

        for step in range(len(selected), target_k):
            best_cand = None
            best_gain = -float("inf")
            j_pos = step
            g_j = gamma[j_pos]

            for cand_idx in remaining:
                lin_gain = utilities[cand_idx] * g_j
                red_penalty = 0.0
                if objective_type == ObjectiveType.PAIRWISE_DIVERSITY and diversity_lambda > 0.0:
                    for prev_j, prev_idx in enumerate(selected):
                        red_penalty += sim_mat[cand_idx, prev_idx] * g_j * gamma[prev_j]

                marginal_gain = lin_gain - (diversity_lambda * red_penalty)
                if marginal_gain > best_gain:
                    best_gain = marginal_gain
                    best_cand = cand_idx

            if best_cand is not None:
                selected.append(best_cand)
                remaining.remove(best_cand)

        obj_val = evaluate_ranking_objective(
            ranking_indices=selected,
            utilities=utilities,
            discounts=gamma,
            objective_type=objective_type,
            similarity_matrix=sim_mat,
            diversity_lambda=diversity_lambda,
        )

        details = {
            "algorithm": "Greedy Submodular Approximation",
            "candidate_count": n,
            "target_k": target_k,
        }
        return selected, obj_val, SolverStatus.APPROXIMATE, details
