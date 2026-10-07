from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for health check response."""

    status: str = Field(..., description="Operational status of the API", examples=["ok"])
    service: str = Field(..., description="Service identifier", examples=["researchmind-api"])
    version: str = Field(..., description="Service semantic version", examples=["1.0.0"])
    database: Optional[dict] = Field(default=None, description="Database health status and latency")


class RootResponse(BaseModel):
    """Schema for root endpoint response."""

    message: str = Field(..., description="Informational message", examples=["ResearchMind AI API is running"])
