"""
Observability and Telemetry API endpoints for ResearchMind AI (Phase 9).

Provides real-time system metrics:
- GET /api/v1/observability/metrics: Latencies, error counts, cache hit rate, solver statuses
"""

from typing import Any, Dict
from fastapi import APIRouter

from app.observability.metrics import get_metrics_collector

router = APIRouter(prefix="/observability", tags=["Observability"])


@router.get(
    "/metrics",
    summary="Get System Observability Metrics",
    description="Retrieve latency percentiles, upstream provider errors, cache hit rate, and solver statuses.",
)
def get_metrics() -> Dict[str, Any]:
    collector = get_metrics_collector()
    return collector.get_summary()
