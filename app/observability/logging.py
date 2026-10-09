"""
Structured logging module for ResearchMind AI pipeline telemetry.

Emits machine-readable JSON logs for production observability, containing:
- request_id
- run_id
- query (redacted)
- source retrieval metadata & errors
- fine-grained stage latencies (BM25, SBERT, RRF, Cross-Encoder, Buildability, Utility, Optimization, Verification, Total)
- result counts
- error summaries
"""

import json
import logging
from typing import Any, Dict, List, Optional

from app.observability.context import get_current_request_id
from app.observability.redaction import redact_sensitive_dict, redact_sensitive_text

logger = logging.getLogger("researchmind.pipeline")


def log_pipeline_telemetry(
    query: str,
    request_id: Optional[str] = None,
    run_id: Optional[str] = None,
    sources: Optional[List[str]] = None,
    provider_status: Optional[Dict[str, str]] = None,
    cached_status: Optional[Dict[str, bool]] = None,
    papers_collected: int = 0,
    result_count: int = 0,
    candidate_count: int = 0,
    latencies_ms: Optional[Dict[str, float]] = None,
    optimization_info: Optional[Dict[str, Any]] = None,
    errors: Optional[List[str]] = None,
    level: int = logging.INFO,
) -> Dict[str, Any]:
    """
    Constructs and emits a structured production telemetry log entry.
    Ensures all query strings and metadata are redacted of any secret credentials.
    """
    active_req_id = request_id or get_current_request_id() or "unknown_req_id"
    cleaned_query = redact_sensitive_text(query)

    latencies = latencies_ms or {}
    # Round latencies for clean JSON presentation
    clean_latencies = {
        k: round(float(v), 2) for k, v in latencies.items() if v is not None
    }

    payload: Dict[str, Any] = {
        "event": "research_pipeline_execution",
        "request_id": active_req_id,
        "run_id": run_id,
        "query": cleaned_query,
        "source_retrieval": {
            "requested_sources": sources or [],
            "provider_status": provider_status or {},
            "cached": cached_status or {},
            "papers_collected": papers_collected,
        },
        "latencies_ms": {
            "source_retrieval": clean_latencies.get("source_retrieval_latency")
            or clean_latencies.get("source_latency", 0.0),
            "normalization": clean_latencies.get("normalization_latency", 0.0),
            "retrieval": clean_latencies.get("retrieval_latency", 0.0),
            "bm25": clean_latencies.get("bm25_latency", 0.0),
            "sbert": clean_latencies.get("sbert_latency", 0.0),
            "rrf": clean_latencies.get("rrf_latency", 0.0),
            "cross_encoder": clean_latencies.get("cross_encoder_latency")
            or clean_latencies.get("reranking_latency", 0.0),
            "buildability": clean_latencies.get("buildability_latency", 0.0),
            "utility_calculation": clean_latencies.get("utility_calculation_latency", 0.0),
            "optimization": clean_latencies.get("optimization_latency", 0.0),
            "verification": clean_latencies.get("verification_latency", 0.0),
            "total_request": clean_latencies.get("total_request_latency", 0.0),
        },
        "result_count": result_count,
        "candidate_count": candidate_count,
        "optimization": optimization_info or {},
        "errors": [redact_sensitive_text(e) for e in (errors or [])],
    }

    # Perform recursive safety pass
    safe_payload = redact_sensitive_dict(payload)

    # Emit JSON log entry
    log_line = json.dumps(safe_payload, default=str)
    logger.log(level, log_line)

    return safe_payload
