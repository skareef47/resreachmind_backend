"""
Observability package for ResearchMind AI.
"""

from app.observability.context import get_current_request_id, set_current_request_id
from app.observability.logging import log_pipeline_telemetry
from app.observability.metrics import MetricsCollector, get_metrics_collector
from app.observability.middleware import RequestLoggingMiddleware
from app.observability.redaction import (
    RedactingLogFilter,
    redact_sensitive_dict,
    redact_sensitive_text,
)

__all__ = [
    "MetricsCollector",
    "get_metrics_collector",
    "RequestLoggingMiddleware",
    "get_current_request_id",
    "set_current_request_id",
    "log_pipeline_telemetry",
    "redact_sensitive_text",
    "redact_sensitive_dict",
    "RedactingLogFilter",
]
