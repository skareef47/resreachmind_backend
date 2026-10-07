"""
Observability and Telemetry metrics collector for ResearchMind AI (Phase 9).

Tracks:
- source latency
- retrieval latency
- reranking latency
- optimization latency
- cache hit rate
- provider errors
- candidate counts
- solver status
"""

from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger("researchmind.observability.metrics")


class MetricsCollector:
    """Singleton metrics collector for real-time latency and reliability observability."""

    def __init__(self) -> None:
        self.start_time: float = time.time()
        self.latencies: Dict[str, List[float]] = {
            "source_latency": [],
            "retrieval_latency": [],
            "reranking_latency": [],
            "optimization_latency": [],
        }
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.provider_errors: Dict[str, Dict[str, int]] = {}
        self.candidate_counts: List[int] = []
        self.solver_statuses: Dict[str, int] = {}
        self.total_search_requests: int = 0

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

    def record_provider_error(self, provider: str, error_type: str) -> None:
        """Record upstream scholarly provider error."""
        if provider not in self.provider_errors:
            self.provider_errors[provider] = {}
        self.provider_errors[provider][error_type] = (
            self.provider_errors[provider].get(error_type, 0) + 1
        )

    def record_candidate_count(self, count: int) -> None:
        """Record candidate pool count from a search run."""
        self.candidate_counts.append(count)

    def record_solver_status(self, solver_name: str, status: str) -> None:
        """Record global optimization solver termination status."""
        key = f"{solver_name}:{status}"
        self.solver_statuses[key] = self.solver_statuses.get(key, 0) + 1

    def increment_search_requests(self) -> None:
        """Increment total search requests processed."""
        self.total_search_requests += 1

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
        """Generate comprehensive observability report."""
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
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "total": total_cache,
                "hit_rate": hit_rate,
            },
            "latencies": {
                cat: self._calculate_stats(vals) for cat, vals in self.latencies.items()
            },
            "provider_errors": self.provider_errors,
            "candidate_pool_stats": cand_stats,
            "solver_statuses": self.solver_statuses,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def clear(self) -> None:
        """Reset metrics (useful for isolated testing)."""
        self.latencies = {
            "source_latency": [],
            "retrieval_latency": [],
            "reranking_latency": [],
            "optimization_latency": [],
        }
        self.cache_hits = 0
        self.cache_misses = 0
        self.provider_errors.clear()
        self.candidate_counts.clear()
        self.solver_statuses.clear()
        self.total_search_requests = 0


# Global singleton instance
_GLOBAL_METRICS = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Dependency provider for global metrics collector."""
    return _GLOBAL_METRICS
