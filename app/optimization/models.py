from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.feedback.models import FeedbackItem
from app.optimization.constraints import SearchConstraints
from app.optimization.utility import UtilityWeights
from app.preferences.models import UserPreferences
from app.schemas.retrieval import RetrievedCandidate


class SolverType(str, Enum):
    """Available optimization solvers."""

    LINEAR_ASSIGNMENT = "linear_assignment"  # Hungarian algorithm / linear sum assignment
    ILP_DIVERSITY = "ilp_diversity"          # Mixed-Integer Linear Program for pairwise interactions & diversity
    EXHAUSTIVE = "exhaustive"                # Exhaustive permutation enumeration (ground truth for small N)
    GREEDY = "greedy"                        # Greedy submodular approximation baseline


class ObjectiveType(str, Enum):
    """Mathematical formulation of the ranking objective."""

    LINEAR_POSITIONAL = "linear_positional"      # Sum of position-discounted individual utilities
    PAIRWISE_DIVERSITY = "pairwise_diversity"    # Utility plus pairwise redundancy penalization / diversity


class SolverStatus(str, Enum):
    """Execution status returned by optimization solver."""

    OPTIMAL = "optimal"
    APPROXIMATE = "approximate"
    INFEASIBLE = "infeasible"
    FAILED = "failed"


class VerificationStatus(str, Enum):
    """Verification outcome status."""

    VERIFIED_OPTIMAL = "verified_optimal"
    VERIFIED_FEASIBLE = "verified_feasible"
    APPROXIMATION_GAP_RECORDED = "approximation_gap_recorded"
    CONSTRAINT_VIOLATION = "constraint_violation"
    VERIFICATION_FAILED = "verification_failed"


class OptimizationProblem(BaseModel):
    """
    Formal ranking optimization problem:
    R* = argmax_{R in S_n} U(R | q, D, F, P, C)
    
    Guaranteed ONLY globally optimal over the retrieved candidate set D
    under the explicitly defined utility U and constraints C.
    """

    candidates: List[RetrievedCandidate] = Field(..., description="Retrieved candidate documents D")
    k: int = Field(default=10, ge=1, description="Size of target ranked list R to return")
    objective_type: ObjectiveType = Field(
        default=ObjectiveType.LINEAR_POSITIONAL,
        description="Type of objective function to maximize",
    )
    diversity_lambda: float = Field(
        default=0.2, ge=0.0, description="Redundancy penalty multiplier for pairwise diversity"
    )
    preferences: Optional[UserPreferences] = Field(default=None, description="Active user preferences P")
    feedback_items: Optional[List[FeedbackItem]] = Field(default=None, description="Active user feedback F")
    constraints: Optional[SearchConstraints] = Field(default=None, description="Active search constraints C")
    weights: Optional[UtilityWeights] = Field(default=None, description="Utility function weights")
    solver_type: Optional[SolverType] = Field(
        default=None, description="Optional explicit solver override (e.g. for testing or baselines)"
    )


class VerificationReport(BaseModel):
    """
    Independent audit report validating optimization outcome.
    
    Checks:
    - Recomputed objective vs solver reported objective
    - Hard constraint compliance
    - Exhaustive comparison on small candidate sets
    - Exact approximation gap: Delta U = U(R_exact) - U(R_approx)
    """

    is_verified: bool = Field(..., description="Whether verification succeeded without errors")
    verification_status: VerificationStatus = Field(..., description="Formal verification status")
    solver_status: SolverStatus = Field(..., description="Solver execution status")
    candidate_count: int = Field(..., description="Total candidate count evaluated")
    k: int = Field(..., description="Target rank list length")
    solver_used: SolverType = Field(..., description="Solver employed")
    objective_type: ObjectiveType = Field(..., description="Objective function formulation")
    reported_objective: float = Field(..., description="Objective value reported by solver")
    recomputed_objective: float = Field(..., description="Independently recomputed objective value")
    is_objective_match: bool = Field(..., description="Whether reported and recomputed objectives match")
    constraints_satisfied: bool = Field(..., description="Whether all hard constraints and exclusions hold")
    constraint_violations: List[str] = Field(
        default_factory=list, description="Violation messages if constraints failed"
    )
    exhaustive_comparison_performed: bool = Field(
        default=False, description="Whether exhaustive enumeration baseline was computed"
    )
    exact_reference_objective: Optional[float] = Field(
        default=None, description="True global maximum objective from exhaustive enumeration"
    )
    approximation_gap: Optional[float] = Field(
        default=None,
        description="Approximation gap: Delta U = U(R_exact) - U(R_approx) (>= 0.0)",
    )
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional solver telemetry")


class OptimizationResult(BaseModel):
    """Optimal ranking result with formal verification telemetry."""

    ranked_candidates: List[RetrievedCandidate] = Field(
        ..., description="Globally optimal or verified approximate ranking R*"
    )
    objective_value: float = Field(..., description="Final objective function value achieved")
    solver_used: SolverType = Field(..., description="Algorithm employed to obtain ranking")
    solver_status: SolverStatus = Field(..., description="Status of solver convergence")
    verification: VerificationReport = Field(..., description="Independent verification audit")
    guarantee_scope: str = Field(
        default="globally optimal over the retrieved candidate set under the explicitly defined utility and constraints",
        description="Formal epistemic boundary of the optimality guarantee",
    )
