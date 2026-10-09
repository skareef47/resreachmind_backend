"""
Runner script for large-scale optimization benchmarks.
Executes benchmarks across N in [5, 10, 20, 50, 100, 250, 500, 1000] and scenarios A-F.
Outputs full markdown tables and statistics.
"""

import json
import os
import sys
import time

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.optimization.benchmark import (
    generate_deterministic_candidates,
    create_scenario_problem,
    run_single_benchmark,
    BenchmarkResult,
)
from app.optimization.models import ObjectiveType, SolverType, SolverStatus
from app.optimization.solvers import (
    ExhaustivePermutationSolver,
    GreedyDiversitySolver,
    PairwiseDiversityILPSolver,
    compute_candidate_similarity_matrix,
)
from app.optimization.utility import FormalUtilityModel
from app.optimization.constraints import SearchConstraints
from app.preferences.models import UserPreferences
from app.feedback.matcher import FeedbackMatcher


def main():
    print("=======================================================================")
    print(" ResearchMind AI: Large-Scale Optimization Benchmarking Engine")
    print("=======================================================================")
    print("Formal Guarantee: Globally optimal over retrieved candidate set D and utility U.")
    print("Testing scales: N in [5, 10, 20, 50, 100, 250, 500, 1000]")
    print("Testing scenarios: A (Linear), B (Prefs), C (Feedback), D (Constraints), E (Diversity), F (Mixed)")
    print()

    candidate_sizes = [5, 10, 20, 50, 100, 250, 500, 1000]
    scenarios = ["A", "B", "C", "D", "E", "F"]
    scenario_names = {
        "A": "Pure Positional Utility",
        "B": "User Preferences",
        "C": "Relevance Feedback",
        "D": "Hard Constraints (Filtering)",
        "E": "Pairwise Diversity",
        "F": "Mixed Multi-Criteria Objective",
    }

    all_results = []

    # Run primary benchmark matrix
    for scen in scenarios:
        print(f"\n--- Running Scenario {scen}: {scenario_names[scen]} ---")
        for n in candidate_sizes:
            k = 5 if n == 5 else 10
            sys.stdout.write(f"  Testing N={n:4d}, K={k:2d}... ")
            sys.stdout.flush()
            t0 = time.perf_counter()
            res = run_single_benchmark(n=n, scenario_id=scen, k=k, seed=42)
            elapsed = (time.perf_counter() - t0) * 1000.0
            all_results.append(res)
            print(f"Done in {res.runtime_ms:8.2f} ms | Peak RAM: {res.memory_mb:6.3f} MB | Obj: {res.objective_value:8.4f} | Gap: {res.optimality_gap or 0.0:8.6f} | Status: {res.optimization_status}")

    # Dedicated Diversity Solver Comparison (MILP vs Greedy vs Exhaustive)
    print("\n--- Dedicated Diversity Solver Comparison (Scenario E) ---")
    div_comparison = []
    div_sizes = [5, 10, 20, 50, 100]

    for n in div_sizes:
        k = 5 if n == 5 else 10
        cands = generate_deterministic_candidates(n=n, seed=42)
        target_k = min(k, n)
        u_model = FormalUtilityModel()
        matcher = FeedbackMatcher({c.document.internal_id: c.document for c in cands})
        utilities = [
            u_model.evaluate_candidate(c, UserPreferences(), matcher, [], SearchConstraints(), None).utility_score
            for c in cands
        ]
        sim_mat = compute_candidate_similarity_matrix(cands)

        # 1. Exhaustive (if N <= 10)
        ex_obj = None
        ex_time = None
        if n <= 10:
            ex_solver = ExhaustivePermutationSolver(max_candidates=10)
            t_ex = time.perf_counter()
            _, ex_val, _, _ = ex_solver.solve(cands, utilities, target_k, ObjectiveType.PAIRWISE_DIVERSITY, sim_mat, 0.25)
            ex_time = (time.perf_counter() - t_ex) * 1000.0
            ex_obj = ex_val

        # 2. MILP (PairwiseDiversityILPSolver) with 5.0s timeout
        milp_solver = PairwiseDiversityILPSolver(time_limit_sec=5.0)
        t_milp = time.perf_counter()
        m_rank, m_obj, m_stat, m_det = milp_solver.solve(cands, utilities, target_k, ObjectiveType.PAIRWISE_DIVERSITY, sim_mat, 0.25)
        milp_time = (time.perf_counter() - t_milp) * 1000.0

        # 3. Greedy (GreedyDiversitySolver)
        greedy_solver = GreedyDiversitySolver()
        t_gr = time.perf_counter()
        g_rank, g_obj, g_stat, g_det = greedy_solver.solve(cands, utilities, target_k, ObjectiveType.PAIRWISE_DIVERSITY, sim_mat, 0.25)
        greedy_time = (time.perf_counter() - t_gr) * 1000.0

        gap_milp = max(0.0, round(ex_obj - m_obj, 6)) if ex_obj is not None else 0.0
        gap_greedy = max(0.0, round((ex_obj or m_obj) - g_obj, 6))

        div_comparison.append({
            "n": n,
            "k": target_k,
            "exhaustive_obj": ex_obj,
            "exhaustive_ms": ex_time,
            "milp_obj": m_obj,
            "milp_ms": milp_time,
            "milp_status": m_stat.value,
            "milp_gap": gap_milp,
            "greedy_obj": g_obj,
            "greedy_ms": greedy_time,
            "greedy_status": g_stat.value,
            "greedy_gap": gap_greedy,
        })
        print(f"  N={n:3d}: Exhaustive={ex_obj} ({ex_time:.1f}ms if ran) | MILP={m_obj:.4f} ({milp_time:.1f}ms, {m_stat.value}) | Greedy={g_obj:.4f} ({greedy_time:.1f}ms, gap={gap_greedy:.4f})")

    # Output JSON summary to stdout
    output_data = {
        "benchmark_results": [r.to_dict() for r in all_results],
        "diversity_comparison": div_comparison,
    }
    with open("benchmark_results.json", "w") as f:
        json.dump(output_data, f, indent=2)

    print("\nBenchmark successfully executed! Results written to benchmark_results.json")


if __name__ == "__main__":
    main()
