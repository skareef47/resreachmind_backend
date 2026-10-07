from collections import defaultdict
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.api import api_router
from app.config import get_settings
from app.db.session import init_db
from app.schemas.common import RootResponse

settings = get_settings()

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("researchmind")


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """In-memory rate limiter per IP address respecting configured thresholds."""

    def __init__(self, app: FastAPI, max_requests_per_minute: int = 120):
        super().__init__(app)
        self.max_requests = max_requests_per_minute
        self.window_seconds = 60.0
        self.requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if self.max_requests <= 0 or path in [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/health",
            "/api/v1/health/db",
            "/",
        ]:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - self.window_seconds

        # Clean timestamps older than window
        timestamps = [t for t in self.requests[client_ip] if t > window_start]
        if len(timestamps) >= self.max_requests:
            retry_after = max(1, int(timestamps[0] + self.window_seconds - now) + 1)
            logger.warning("Rate limit exceeded for IP %s on path %s", client_ip, path)
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please try again later."},
                headers={"Retry-After": str(retry_after)},
            )

        timestamps.append(now)
        self.requests[client_ip] = timestamps
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Handle application startup and shutdown lifecycle events."""
    logger.info(
        "Starting %s API v%s (environment: %s, debug: %s)",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.ENVIRONMENT,
        settings.DEBUG,
    )
    logger.info("Configured CORS origins: %s", settings.CORS_ORIGINS)

    # Initialize persistence layer schema
    try:
        await init_db()
        logger.info("Database schema initialized.")
    except Exception as exc:
        logger.error("Failed to initialize database schema: %s", exc)

    yield
    logger.info("Shutting down %s API", settings.APP_NAME)


def create_application() -> FastAPI:
    """Initialize and configure the FastAPI application instance."""
    app = FastAPI(
        title="ResearchMind AI API",
        version=settings.APP_VERSION,
        description="FastAPI backend foundation for ResearchMind AI.",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS for frontend access
    origins = settings.CORS_ORIGINS if settings.CORS_ORIGINS else ["http://localhost:3000"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting middleware
    app.add_middleware(
        RateLimiterMiddleware,
        max_requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
    )

    # Root endpoint
    @app.get(
        "/",
        response_model=RootResponse,
        tags=["Root"],
        summary="Root Endpoint",
        description="Check if the ResearchMind AI API is running.",
    )
    def root() -> RootResponse:
        return RootResponse(message="ResearchMind AI API is running")

    # Register API v1 routes
    app.include_router(api_router, prefix=settings.API_PREFIX)

    return app


app = create_application()
