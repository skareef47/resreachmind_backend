from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

client = TestClient(app)
settings = get_settings()


def test_root_endpoint() -> None:
    """Test the root endpoint returns 200 and running message."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "ResearchMind AI API is running"


def test_health_endpoint() -> None:
    """Test the /api/v1/health endpoint meets Phase 1 specification."""
    response = client.get(f"{settings.API_PREFIX}/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "researchmind-api"
    assert "version" in data
    assert data["version"] == settings.APP_VERSION
    assert len(data["version"]) > 0


def test_openapi_docs_accessible() -> None:
    """Verify OpenAPI specification and docs endpoints are generated."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "ResearchMind AI API"
    assert f"{settings.API_PREFIX}/health" in schema["paths"]
    assert f"{settings.API_PREFIX}/search" in schema["paths"]
