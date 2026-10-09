"""
Observability and Telemetry metrics collector for ResearchMind AI.

Tracks:
- Total HTTP request count, success count, error count
- Granular stage latencies:
  * total_request_latency
  * source_retrieval_latency
  * normalization_latency
  * retrieval_latency
  * bm25_latency
  * sbert_latency
  * rrf_latency
  * cross_encoder_latency
  * buildability_latency
  * utility_calculation_latency
  * optimization_latency
  * verification_latency
- Source failure counters (Semantic Scholar vs. OpenAlex)
- Optimization and verification failure counters
- Cache hits, misses, hit rate
- Solver status telemetry
- Candidate pool sizes
"""

from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger("researchmind.observability.metrics")

ALL_LATENCY_CATEGORIES = [
    "total_request_latency",
    "source_retrieval_latency",
    "source_latency",  # alias for backward-compatibility
    "normalization_latency",
    "retrieval_latency",
    "bm25_latency",
    "sbert_latency",
    "rrf_latency",
    "cross_encoder_latency",
    "reranking_latency",  # alias
    "buildability_latency",
    "utility_calculation_latency",
    "optimization_latency",
    "verification_latency",
]


class MetricsCollector:
    """Singleton metrics collector for real-time latency, error, and reliability observability."""

    def __init__(self) -> None:
        self.start_time: float = time.time()
        self.request_count: int = 0
        self.success_count: int = 0
        self.error_count: int = 0
        self.total_search_requests: int = 0

        self.latencies: Dict[str, List[float]] = {
            cat: [] for cat in ALL_LATENCY_CATEGORIES
        }

        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.provider_errors: Dict[str, Dict[str, int]] = {}
        self.optimization_failures: int = 0
        self.verification_failures: int = 0

        self.candidate_counts: List[int] = []
        self.solver_statuses: Dict[str, int] = {}

    def record_request(self, success: bool = True, status_code: int = 200) -> None:
        """Record an incoming HTTP request and its high-level outcome."""
        self.request_count += 1
        if success and status_code < 400:
            self.success_count += 1
        else:
            self.error_count += 1

    def increment_search_requests(self) -> None:
        """Increment total search queries processed."""
        self.total_search_requests += 1

    def record_latency(self, category: str, latency_ms: float) -> None:
        """Record an operation execution latency in milliseconds."""
        if category not in self.latencies:
            self.latencies[category] = []
        self.latencies[category].append(round(latency_ms, 2))

    def record_cache_access(self, is_hit: bool) -> None:
        """Record cache query outcome."""
        if is_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def record_source_failure(self, provider: str, error_type: str) -> None:
        """Record upstream scholarly provider failure."""
        if provider not in self.provider_errors:
            self.provider_errors[provider] = {}
        self.provider_errors[provider][error_type] = (
            self.provider_errors[provider].get(error_type, 0) + 1
        )

    def record_provider_error(self, provider: str, error_type: str) -> None:
        """Alias for backward-compatibility with earlier code."""
        self.record_source_failure(provider, error_type)

    def record_optimization_failure(self) -> None:
        """Record global optimization failure or fallback."""
        self.optimization_failures += 1

    def record_verification_failure(self) -> None:
        """Record formal optimality verification failure or discrepancy."""
        self.verification_failures += 1

    def record_candidate_count(self, count: int) -> None:
        """Record candidate pool count from a search run."""
        self.candidate_counts.append(count)

    def record_solver_status(self, solver_name: str, status: str) -> None:
        """Record global optimization solver termination status."""
        key = f"{solver_name}:{status}"
        self.solver_statuses[key] = self.solver_statuses.get(key, 0) + 1

    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Compute summary statistics for a latency series."""
        if not values:
            return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        arr = np.array(values)
        return {
            "count": len(values),
            "mean_ms": float(round(float(np.mean(arr)), 2)),
            "p50_ms": float(round(float(np.percentile(arr, 50)), 2)),
            "p95_ms": float(round(float(np.percentile(arr, 95)), 2)),
            "max_ms": float(round(float(np.max(arr)), 2)),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Generate comprehensive, non-sensitive observability report."""
        total_cache = self.cache_hits + self.cache_misses
        hit_rate = round(self.cache_hits / total_cache, 4) if total_cache > 0 else 0.0

        cand_stats = {}
        if self.candidate_counts:
            cand_stats = {
                "total_runs": len(self.candidate_counts),
                "mean_candidates": float(round(float(np.mean(self.candidate_counts)), 2)),
                "min_candidates": int(min(self.candidate_counts)),
                "max_candidates": int(max(self.candidate_counts)),
            }

        return {
            "uptime_seconds": round(time.time() - self.start_time, 2),
            "total_search_requests": self.total_search_requests,
            "requests": {
                "total": self.request_count,
                "success": self.success_count,
                "error": self.error_count,
                "search_requests": self.total_search_requests,
            },
            "failures": {
                "source_failures": self.provider_errors,
                "optimization_failures": self.optimization_failures,
                "verification_failures": self.verification_failures,
            },
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "total": total_cache,
                "hit_rate": hit_rate,
            },
            "latencies": {
                cat: self._calculate_stats(vals)
                for cat, vals in self.latencies.items()
                if vals  # Only show categories with recorded samples or summary
            },
            "all_latencies_summary": {
                cat: self._calculate_stats(vals)
                for cat, vals in self.latencies.items()
            },
            "provider_errors": self.provider_errors,
            "candidate_pool_stats": cand_stats,
            "solver_statuses": self.solver_statuses,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def clear(self) -> None:
        """Reset metrics (useful for isolated testing)."""
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_search_requests = 0
        self.latencies = {cat: [] for cat in ALL_LATENCY_CATEGORIES}
        self.cache_hits = 0
        self.cache_misses = 0
        self.provider_errors.clear()
        self.optimization_failures = 0
        self.verification_failures = 0
        self.candidate_counts.clear()
        self.solver_statuses.clear()


# Global singleton instance
_GLOBAL_METRICS = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Dependency provider for global metrics collector."""
    return _GLOBAL_METRICS
