"""Optimization package: formal utility models, constraints, solvers, global optimizer, and audit verification."""

from app.optimization.constraints import (
    ConstraintContradictionError,
    ConstraintEvaluationResult,
    ConstraintEvaluator,
    SearchConstraints,
    detect_constraint_contradictions,
    validate_constraints,
)
from app.optimization.models import (
    ObjectiveType,
    OptimizationProblem,
    OptimizationResult,
    SolverStatus,
    SolverType,
    VerificationReport,
    VerificationStatus,
)
from app.optimization.optimizer import (
    OPTIMALITY_GUARANTEE_STATEMENT,
    GlobalRankingOptimizer,
)
from app.optimization.solvers import (
    BaseRankingSolver,
    ExhaustivePermutationSolver,
    GreedyDiversitySolver,
    LinearAssignmentSolver,
    PairwiseDiversityILPSolver,
    calculate_position_discounts,
    compute_candidate_similarity_matrix,
    evaluate_ranking_objective,
)
from app.optimization.utility import (
    CandidateEvaluation,
    FormalUtilityModel,
    UtilityComponentScores,
    UtilityRunRecord,
    UtilityRunStore,
    UtilityWeights,
    get_utility_run_store,
)
from app.optimization.verification import OptimizationVerifier

__all__ = [
    # Constraints
    "SearchConstraints",
    "ConstraintEvaluationResult",
    "ConstraintEvaluator",
    "ConstraintContradictionError",
    "detect_constraint_contradictions",
    "validate_constraints",
    # Utility
    "UtilityWeights",
    "UtilityComponentScores",
    "CandidateEvaluation",
    "UtilityRunRecord",
    "UtilityRunStore",
    "FormalUtilityModel",
    "get_utility_run_store",
    # Models
    "SolverType",
    "ObjectiveType",
    "SolverStatus",
    "VerificationStatus",
    "OptimizationProblem",
    "VerificationReport",
    "OptimizationResult",
    # Solvers
    "BaseRankingSolver",
    "LinearAssignmentSolver",
    "PairwiseDiversityILPSolver",
    "ExhaustivePermutationSolver",
    "GreedyDiversitySolver",
    "calculate_position_discounts",
    "compute_candidate_similarity_matrix",
    "evaluate_ranking_objective",
    # Verifier & Optimizer
    "OptimizationVerifier",
    "GlobalRankingOptimizer",
    "OPTIMALITY_GUARANTEE_STATEMENT",
]
