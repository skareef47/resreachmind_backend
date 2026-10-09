from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Response, status

from app.schemas.common import HealthResponse
from app.services.health_service import HealthService

router = APIRouter(tags=["Health"])


def get_health_service() -> HealthService:
    """Dependency provider for HealthService."""
    return HealthService()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API Health Status",
    description="Returns operational status, service name, active version, and database connectivity.",
)
async def get_health(
    service: HealthService = Depends(get_health_service),
) -> HealthResponse:
    """Check and return system health information with database status."""
    base_health = service.get_health()
    from app.db.session import check_db_health
    db_status = await check_db_health()
    return HealthResponse(
        status=base_health.status if db_status.get("status") != "degraded" else "degraded",
        service=base_health.service,
        version=base_health.version,
        database=db_status,
    )


@router.get(
    "/health/live",
    summary="Liveness Probe",
    description="Kubernetes/container liveness probe confirming the application process is running.",
)
def get_liveness() -> dict:
    """Liveness probe returning 200 OK if process is responding."""
    return {
        "status": "ok",
        "live": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/health/ready",
    summary="Readiness Probe",
    description="Kubernetes/container readiness probe checking database connectivity and readiness to receive traffic.",
)
async def get_readiness(response: Response) -> dict:
    """Readiness probe checking backing services connectivity."""
    from app.db.session import check_db_health
    db_status = await check_db_health()
    is_ready = db_status.get("status") in ["connected", "healthy"]

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "healthy" if is_ready else "degraded",
        "ready": is_ready,
        "database": db_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/health/db",
    summary="Database Direct Health Status",
    description="Pings database directly and reports connection latency and dialect status.",
)
async def get_db_health() -> dict:
    from app.db.session import check_db_health
    return await check_db_health()
