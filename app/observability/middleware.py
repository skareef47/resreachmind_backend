"""
Request logging and correlation ID middleware for ResearchMind AI.

- Assigns or extracts X-Request-ID for distributed tracing.
- Measures total HTTP request latency and records X-Response-Time-Ms header.
- Records global request counters (total, success, error).
- Attaches request ID to async ContextVar.
- Emits structured access logs with sensitive data redacted.
"""

import logging
import time
import uuid
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.observability.context import set_current_request_id
from app.observability.metrics import get_metrics_collector
from app.observability.redaction import redact_sensitive_text

logger = logging.getLogger("researchmind.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """FastAPI/Starlette middleware for request correlation, metrics, and latency measurement."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Extract or generate X-Request-ID
        incoming_id = request.headers.get("x-request-id") or request.headers.get("X-Request-ID")
        if incoming_id and incoming_id.strip():
            request_id = incoming_id.strip()[:64]
        else:
            request_id = uuid.uuid4().hex

        # Store in ContextVar and request state
        set_current_request_id(request_id)
        request.state.request_id = request_id

        metrics = get_metrics_collector()
        start_time = time.perf_counter()

        status_code = 500
        response = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as exc:
            # Record unhandled failure
            metrics.record_request(success=False, status_code=500)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            metrics.record_latency("total_request_latency", latency_ms)
            logger.error(
                "Unhandled error processing request [req_id=%s] %s %s: %s",
                request_id,
                request.method,
                redact_sensitive_text(str(request.url.path)),
                exc,
                exc_info=True,
            )
            raise exc
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # Attach tracing headers if response exists
            if response is not None:
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Response-Time-Ms"] = f"{latency_ms:.2f}"
                metrics.record_latency("total_request_latency", latency_ms)
                metrics.record_request(success=(status_code < 400), status_code=status_code)

            # Avoid spamming logs on static docs/favicon endpoints unless debug
            path = request.url.path
            if path not in ["/openapi.json", "/favicon.ico"]:
                client_ip = request.client.host if request.client else "unknown"
                logger.info(
                    "HTTP %s %s | status=%s | latency=%.2fms | req_id=%s | ip=%s",
                    request.method,
                    redact_sensitive_text(path),
                    status_code,
                    latency_ms,
                    request_id,
                    client_ip,
                )
