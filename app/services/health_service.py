from typing import Optional

from app.config import Settings, get_settings
from app.schemas.common import HealthResponse


class HealthService:
    """Service handling application health check verification."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def get_health(self) -> HealthResponse:
        """Return the health status of the application."""
        return HealthResponse(
            status="ok",
            service="researchmind-api",
            version=self.settings.APP_VERSION,
        )
