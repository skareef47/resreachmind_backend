from fastapi import APIRouter, Depends

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
    "/health/db",
    summary="Database Direct Health Status",
    description="Pings database directly and reports connection latency and dialect status.",
)
async def get_db_health() -> dict:
    from app.db.session import check_db_health
    return await check_db_health()

