"""
Comprehensive Test Suite for Production-Grade Observability in ResearchMind AI.

Validates:
1. Health endpoints (/health, /health/live, /health/ready)
2. Request ID generation, propagation in headers, and inclusion in responses
3. Run ID tracking across search and optimization pipelines
4. Structured logging containing all 11+ granular latencies and metadata
5. External source failure isolation (Semantic Scholar failure does not crash OpenAlex)
6. Optimization failure tracking and graceful fallback
7. Latency and operational metrics recording in /api/v1/metrics
8. Strict secret and credential redaction across text, dictionaries, and logs
"""

import json
import logging
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
from fastapi.testclient import TestClient

from app.api.search import get_search_adapters
from app.config import get_settings
from app.ingestion.cache import get_cache
from app.main import app
from app.observability.context import get_current_request_id, set_current_request_id
from app.observability.metrics import get_metrics_collector
from app.observability.redaction import (
    RedactingLogFilter,
    redact_sensitive_dict,
    redact_sensitive_text,
)
from app.schemas.source import SourcePaper

client = TestClient(app)
settings = get_settings()


@pytest.fixture(autouse=True)
def cleanup_overrides_and_cache():
    """Clear cache and dependency overrides for test isolation."""
    import asyncio
    asyncio.run(get_cache().clear())
    yield
    app.dependency_overrides.clear()


def test_health_endpoints_live_and_ready() -> None:
    """Validate /health, /health/live, and /health/ready endpoints."""
    # 1. Main health
    resp = client.get(f"{settings.API_PREFIX}/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ["ok", "healthy", "degraded"]
    assert data["service"] == "researchmind-api"
    assert "database" in data

    # 2. Liveness probe (/health/live)
    live_resp = client.get(f"{settings.API_PREFIX}/health/live")
    assert live_resp.status_code == 200
    live_data = live_resp.json()
    assert live_data["status"] == "ok"
    assert live_data["live"] is True
    assert "timestamp" in live_data

    # 3. Readiness probe (/health/ready)
    ready_resp = client.get(f"{settings.API_PREFIX}/health/ready")
    assert ready_resp.status_code in [200, 503]
    ready_data = ready_resp.json()
    assert "ready" in ready_data
    assert "database" in ready_data
    assert "timestamp" in ready_data


def test_request_id_propagation_and_custom_header() -> None:
    """Verify X-Request-ID propagation, generation, and response time header."""
    # Custom request ID
    custom_id = "test-req-trace-987654321"
    resp = client.get(f"{settings.API_PREFIX}/health", headers={"X-Request-ID": custom_id})
    assert resp.status_code == 200
    assert resp.headers.get("x-request-id") == custom_id
    assert "x-response-time-ms" in resp.headers
    latency = float(resp.headers["x-response-time-ms"])
    assert latency >= 0.0

    # Auto-generated request ID
    resp_auto = client.get(f"{settings.API_PREFIX}/health")
    assert resp_auto.status_code == 200
    auto_id = resp_auto.headers.get("x-request-id")
    assert auto_id is not None
    assert len(auto_id) >= 16


def test_request_id_and_run_id_in_search_response() -> None:
    """Verify search response returns request_id and run_id."""
    sample_papers = [
        SourcePaper(
            source="openalex",
            source_id="W12345",
            title="Efficient Attention Mechanics in Large Transformers",
            abstract="We present an efficient linear transformer with open source code and dataset.",
            authors=["Vaswani et al."],
            year=2023,
            venue="NeurIPS",
            citation_count=150,
        )
    ]

    with patch("app.api.search.get_search_adapters") as mock_adapters:
        mock_adapter = MagicMock()
        mock_adapter.search = AsyncMock(return_value=sample_papers)
        mock_adapter.close = AsyncMock()
        mock_adapters.return_value = {"openalex": mock_adapter}

        custom_id = "trace-search-xyz-111"
        resp = client.post(
            f"{settings.API_PREFIX}/search",
            headers={"X-Request-ID": custom_id},
            json={
                "query": "efficient attention mechanisms",
                "limit": 5,
                "sources": ["openalex"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert resp.headers.get("x-request-id") == custom_id
        assert data.get("request_id") == custom_id
        assert data.get("run_id") is not None
        assert len(data["run_id"]) > 0


def test_successful_request_logging_and_latencies(caplog: pytest.LogCaptureFixture) -> None:
    """Verify structured pipeline execution log contains all required latencies."""
    sample_papers = [
        SourcePaper(
            source="semantic_scholar",
            source_id="S999",
            title="Diffusion Models for Scientific Discovery",
            abstract="Pretrained models and code benchmarked on synthetic data.",
            authors=["Alice Smith"],
            year=2024,
            venue="ICLR",
            citation_count=45,
        )
    ]

    with patch("app.api.search.get_search_adapters") as mock_adapters:
        mock_adapter = MagicMock()
        mock_adapter.search = AsyncMock(return_value=sample_papers)
        mock_adapter.close = AsyncMock()
        mock_adapters.return_value = {"semantic_scholar": mock_adapter}

        with caplog.at_level(logging.INFO):
            resp = client.post(
                f"{settings.API_PREFIX}/search",
                headers={"X-Request-ID": "log-test-req-777"},
                json={
                    "query": "diffusion models discovery",
                    "limit": 3,
                    "sources": ["semantic_scholar"],
                    "optimize": True,
                },
            )
            assert resp.status_code == 200

        # Find the structured json log entry
        telemetry_entry = None
        for record in caplog.records:
            if "research_pipeline_execution" in record.message:
                try:
                    telemetry_entry = json.loads(record.message)
                    break
                except Exception:
                    pass

        assert telemetry_entry is not None, "Structured pipeline telemetry was not logged"
        assert telemetry_entry["request_id"] == "log-test-req-777"
        assert telemetry_entry["query"] == "diffusion models discovery"
        assert "latencies_ms" in telemetry_entry

        lat = telemetry_entry["latencies_ms"]
        # Verify all 11+ granular latencies exist
        expected_latency_keys = [
            "source_retrieval",
            "normalization",
            "retrieval",
            "bm25",
            "sbert",
            "rrf",
            "cross_encoder",
            "buildability",
            "utility_calculation",
            "optimization",
            "verification",
            "total_request",
        ]
        for key in expected_latency_keys:
            assert key in lat, f"Missing latency key: {key}"
            assert isinstance(lat[key], (int, float))


def test_external_source_failure_isolation() -> None:
    """Verify that a failure in Semantic Scholar does NOT crash OpenAlex."""
    metrics = get_metrics_collector()
    metrics.clear()

    openalex_paper = SourcePaper(
        source="openalex",
        source_id="W555",
        title="Robustness in Distributed Information Systems",
        abstract="A study on fault tolerance.",
        authors=["Charlie Davis"],
        year=2022,
    )

    mock_s2 = MagicMock()
    mock_s2.search = AsyncMock(side_effect=RuntimeError("Semantic Scholar upstream timeout 504"))
    mock_s2.close = AsyncMock()

    mock_oa = MagicMock()
    mock_oa.search = AsyncMock(return_value=[openalex_paper])
    mock_oa.close = AsyncMock()

    app.dependency_overrides[get_search_adapters] = lambda: {
        "semantic_scholar": mock_s2,
        "openalex": mock_oa,
    }

    unique_query = f"fault tolerance isolated query {uuid.uuid4().hex[:8]}"
    resp = client.post(
        f"{settings.API_PREFIX}/search",
        json={
            "query": unique_query,
            "limit": 5,
            "sources": ["semantic_scholar", "openalex"],
        },
    )
    assert resp.status_code == 200, f"Expected 200 OK despite provider error, got {resp.status_code}"
    data = resp.json()

    # Semantic Scholar recorded failure
    assert "error:" in data["provider_status"]["semantic_scholar"]
    # OpenAlex succeeded
    assert data["provider_status"]["openalex"] == "ok"

    # Results still retrieved from OpenAlex!
    assert data["total_results"] >= 1
    assert any(p["source"] == "openalex" for p in data["results"])

    # Metric recorded provider error
    summary = metrics.get_summary()
    assert "semantic_scholar" in summary["failures"]["source_failures"]


def test_optimization_failure_fallback_and_metric() -> None:
    """Verify that optimization exceptions trigger fallback and record optimization failure metric."""
    metrics = get_metrics_collector()
    metrics.clear()

    sample_papers = [
        SourcePaper(
            source="openalex",
            source_id="W100",
            title="Graph Neural Networks for Chemistry",
            abstract="GNNs applied to molecular structure prediction.",
            authors=["Dana Lee"],
            year=2023,
        )
    ]

    with patch("app.api.search.get_search_adapters") as mock_adapters:
        mock_oa = MagicMock()
        mock_oa.search = AsyncMock(return_value=sample_papers)
        mock_oa.close = AsyncMock()
        mock_adapters.return_value = {"openalex": mock_oa}

        with patch("app.optimization.optimizer.GlobalRankingOptimizer.optimize", side_effect=ValueError("Simulated Solver Infeasibility")):
            resp = client.post(
                f"{settings.API_PREFIX}/search",
                json={
                    "query": "molecular graph neural networks",
                    "limit": 3,
                    "optimize": True,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["optimization"]["status"] == "failed"
            assert metrics.optimization_failures >= 1


def test_system_metrics_endpoint_and_latency_recording() -> None:
    """Verify /api/v1/metrics returns system counters and does not leak secrets."""
    metrics = get_metrics_collector()
    metrics.clear()

    # Record some synthetic metrics
    metrics.record_request(success=True, status_code=200)
    metrics.record_latency("source_retrieval_latency", 45.2)
    metrics.record_latency("bm25_latency", 5.1)
    metrics.record_latency("sbert_latency", 12.8)
    metrics.record_latency("total_request_latency", 98.4)
    metrics.record_cache_access(is_hit=True)
    metrics.record_cache_access(is_hit=False)

    resp = client.get(f"{settings.API_PREFIX}/metrics")
    assert resp.status_code == 200
    data = resp.json()

    assert "uptime_seconds" in data
    assert "requests" in data
    assert data["requests"]["total"] >= 1
    assert "failures" in data
    assert "cache" in data
    assert data["cache"]["hits"] >= 1
    assert data["cache"]["misses"] >= 1
    assert "latencies" in data
    assert "total_request_latency" in data["latencies"]
    assert data["latencies"]["total_request_latency"]["count"] >= 1

    # Verify no secrets in metrics output
    metrics_str = json.dumps(data)
    for forbidden in ["api_key", "password", "secret", "bearer", "sk-"]:
        assert forbidden not in metrics_str.lower() or "[redacted]" in metrics_str.lower()


def test_secret_redaction_engine() -> None:
    """Verify sensitive credentials (keys, bearer tokens, passwords) are masked in all layers."""
    # 1. Plain text redaction
    sample_text = (
        "Calling Semantic Scholar with api_key='sk-1234567890abcdef' "
        "and header Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.t-ID "
        "and db_password: my_super_secret_password_123"
    )
    redacted = redact_sensitive_text(sample_text)
    assert "sk-1234567890abcdef" not in redacted
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in redacted
    assert "my_super_secret_password_123" not in redacted
    assert "[REDACTED]" in redacted

    # 2. Dictionary redaction
    raw_dict = {
        "user": "alice",
        "api_key": "secret-api-key-99999",
        "nested": {
            "token": "refresh-token-xyz-88888",
            "password": "p@ssword123!",
            "safe_metric": 42.5,
        },
    }
    cleaned = redact_sensitive_dict(raw_dict)
    assert cleaned["user"] == "alice"
    assert cleaned["api_key"] == "[REDACTED]"
    assert cleaned["nested"]["token"] == "[REDACTED]"
    assert cleaned["nested"]["password"] == "[REDACTED]"
    assert cleaned["nested"]["safe_metric"] == 42.5

    # 3. Log filter interception
    test_logger = logging.getLogger("test.redaction.logger")
    test_logger.setLevel(logging.INFO)
    filter_instance = RedactingLogFilter()
    test_logger.addFilter(filter_instance)

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Connecting with credentials: password=supersecret123",
        args=(),
        exc_info=None,
    )
    filter_instance.filter(record)
    assert "supersecret123" not in record.msg
    assert "[REDACTED]" in record.msg
