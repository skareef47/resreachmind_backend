"""
Observability and Telemetry API endpoints for ResearchMind AI.

Provides real-time system metrics:
- GET /api/v1/metrics: Primary production metrics endpoint
- GET /api/v1/observability/metrics: Observability namespace alias
"""

from typing import Any, Dict
from fastapi import APIRouter

from app.observability.metrics import get_metrics_collector

router = APIRouter(tags=["Observability"])


@router.get(
    "/metrics",
    summary="Get System Observability Metrics",
    description="Retrieve system request counters, granular latency percentiles, error rates, cache statistics, and solver statuses. Redacted of all secrets.",
)
def get_system_metrics() -> Dict[str, Any]:
    """Production metrics endpoint reporting operational performance and error states."""
    collector = get_metrics_collector()
    return collector.get_summary()


@router.get(
    "/observability/metrics",
    summary="Get System Observability Metrics (Alias)",
    description="Alias for /api/v1/metrics for backward compatibility.",
    include_in_schema=False,
)
def get_observability_metrics_alias() -> Dict[str, Any]:
    """Alias returning the same system metrics report."""
    return get_system_metrics()
