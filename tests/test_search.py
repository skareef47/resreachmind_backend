import asyncio
from typing import Dict, List
import pytest
from fastapi.testclient import TestClient

from app.api.search import get_search_adapters
from app.config import get_settings
from app.ingestion.cache import get_cache
from app.main import app
from app.schemas.source import SourcePaper
from app.sources.base import BaseSourceAdapter, SourceTimeoutError

client = TestClient(app)
settings = get_settings()


@pytest.fixture(autouse=True)
def clear_search_cache() -> None:
    """Clear in-memory cache before every test to guarantee test isolation."""
    asyncio.run(get_cache().clear())


class DummySuccessAdapter(BaseSourceAdapter):
    """Mock adapter returning pre-configured papers."""

    def __init__(self, name: str, papers: List[SourcePaper]) -> None:
        super().__init__(base_url="https://mock.local")
        self.source_name = name
        self._papers = papers
        self.call_count = 0

    async def search(self, query: str, limit: int = 10) -> List[SourcePaper]:
        self.call_count += 1
        return self._papers[:limit]


class DummyFailingAdapter(BaseSourceAdapter):
    """Mock adapter raising a configured exception."""

    def __init__(self, name: str, exc: Exception) -> None:
        super().__init__(base_url="https://mock.local")
        self.source_name = name
        self._exc = exc

    async def search(self, query: str, limit: int = 10) -> List[SourcePaper]:
        raise self._exc


@pytest.fixture
def sample_papers() -> Dict[str, List[SourcePaper]]:
    return {
        "semantic_scholar": [
            SourcePaper(
                source="semantic_scholar",
                source_id="s2_1",
                title="Attention Mechanisms in Vision",
                abstract="A study on vision transformers.",
                authors=["Jane Doe"],
                year=2021,
                citation_count=150,
            )
        ],
        "openalex": [
            SourcePaper(
                source="openalex",
                source_id="W12345",
                title="Scaling Vision Transformers",
                abstract="Scaling ViT to billions of parameters.",
                authors=["John Smith"],
                year=2022,
                citation_count=320,
            )
        ],
    }


def test_search_endpoint_success(sample_papers: Dict[str, List[SourcePaper]]) -> None:
    """Verify POST /api/v1/search queries providers, normalizes, deduplicates, and returns canonical ResearchDocuments."""
    mock_adapters = {
        "semantic_scholar": DummySuccessAdapter("semantic_scholar", sample_papers["semantic_scholar"]),
        "openalex": DummySuccessAdapter("openalex", sample_papers["openalex"]),
    }

    app.dependency_overrides[get_search_adapters] = lambda: mock_adapters
    try:
        response = client.post(
            f"{settings.API_PREFIX}/search",
            json={"query": "vision transformers", "limit": 10},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "vision transformers"
        assert data["total_results"] == 2
        assert data["provider_status"] == {
            "semantic_scholar": "ok",
            "openalex": "ok",
        }
        assert len(data["results"]) == 2

        # Check canonical document schema fields
        for doc in data["results"]:
            assert "internal_id" in doc
            assert doc["internal_id"].startswith("doc_")
            assert "title" in doc
            assert "authors" in doc
            assert "provenance" in doc
            assert len(doc["provenance"]) >= 1

        sources = {r["source"] for r in data["results"]}
        assert sources == {"semantic_scholar", "openalex"}

        # Check hybrid retrieval candidates & Cross-Encoder reranking
        assert "candidates" in data
        assert len(data["candidates"]) == 2
        for cand in data["candidates"]:
            assert "document" in cand
            assert "metadata" in cand
            assert "rrf_rank" in cand["metadata"]
            assert "rrf_score" in cand["metadata"]
            # Phase 5 Cross-Encoder metadata
            assert "rerank_metadata" in cand
            assert cand["rerank_metadata"] is not None
            assert "pre_rerank_rank" in cand["rerank_metadata"]
            assert "post_rerank_rank" in cand["rerank_metadata"]
            assert "reranker_score" in cand["rerank_metadata"]
            assert "model_metadata" in cand["rerank_metadata"]
            assert "features" in cand
            assert cand["features"] is not None
            assert "cross_encoder_score" in cand["features"]
    finally:
        app.dependency_overrides.clear()


def test_search_provider_failure_isolation(sample_papers: Dict[str, List[SourcePaper]]) -> None:
    """Verify that if one provider fails/times out, the other provider's results are still returned."""
    mock_adapters = {
        "semantic_scholar": DummyFailingAdapter(
            "semantic_scholar",
            SourceTimeoutError("Upstream gateway timed out", "semantic_scholar"),
        ),
        "openalex": DummySuccessAdapter("openalex", sample_papers["openalex"]),
    }

    app.dependency_overrides[get_search_adapters] = lambda: mock_adapters
    try:
        response = client.post(
            f"{settings.API_PREFIX}/search",
            json={"query": "vision transformers", "limit": 10},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_results"] == 1
        assert data["provider_status"]["semantic_scholar"].startswith("error: SourceTimeoutError")
        assert data["provider_status"]["openalex"] == "ok"
        assert len(data["results"]) == 1
        assert data["results"][0]["source"] == "openalex"
    finally:
        app.dependency_overrides.clear()


def test_search_source_filter(sample_papers: Dict[str, List[SourcePaper]]) -> None:
    """Verify specifying sources only queries the selected providers."""
    mock_adapters = {
        "semantic_scholar": DummySuccessAdapter("semantic_scholar", sample_papers["semantic_scholar"]),
        "openalex": DummySuccessAdapter("openalex", sample_papers["openalex"]),
    }

    app.dependency_overrides[get_search_adapters] = lambda: mock_adapters
    try:
        response = client.post(
            f"{settings.API_PREFIX}/search",
            json={"query": "vision transformers", "limit": 5, "sources": ["openalex"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_results"] == 1
        assert "openalex" in data["provider_status"]
        assert "semantic_scholar" not in data["provider_status"]
        assert data["results"][0]["source"] == "openalex"
    finally:
        app.dependency_overrides.clear()


def test_search_endpoint_caching(sample_papers: Dict[str, List[SourcePaper]]) -> None:
    """Verify repeated search requests hit the cache and do not re-invoke adapters."""
    s2_adapter = DummySuccessAdapter("semantic_scholar", sample_papers["semantic_scholar"])
    oa_adapter = DummySuccessAdapter("openalex", sample_papers["openalex"])
    mock_adapters = {
        "semantic_scholar": s2_adapter,
        "openalex": oa_adapter,
    }

    app.dependency_overrides[get_search_adapters] = lambda: mock_adapters
    try:
        # First request (cache miss)
        r1 = client.post(f"{settings.API_PREFIX}/search", json={"query": "diffusion models", "limit": 5})
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["cached"] == {"semantic_scholar": False, "openalex": False}
        assert s2_adapter.call_count == 1
        assert oa_adapter.call_count == 1

        # Second request (cache hit)
        r2 = client.post(f"{settings.API_PREFIX}/search", json={"query": "diffusion models", "limit": 5})
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["cached"] == {"semantic_scholar": True, "openalex": True}
        # Call count must NOT increase
        assert s2_adapter.call_count == 1
        assert oa_adapter.call_count == 1
    finally:
        app.dependency_overrides.clear()


def test_search_invalid_requests() -> None:
    """Verify validation rejects empty queries and out-of-bound limits."""
    # Empty query
    r_empty = client.post(f"{settings.API_PREFIX}/search", json={"query": "", "limit": 10})
    assert r_empty.status_code == 422

    # Whitespace query
    r_ws = client.post(f"{settings.API_PREFIX}/search", json={"query": "   ", "limit": 10})
    assert r_ws.status_code == 422

    # Limit too small (limit < 1)
    r_zero = client.post(f"{settings.API_PREFIX}/search", json={"query": "ai", "limit": 0})
    assert r_zero.status_code == 422

    # Limit too large (limit > 100)
    r_large = client.post(f"{settings.API_PREFIX}/search", json={"query": "ai", "limit": 101})
    assert r_large.status_code == 422

    # Unsupported source
    r_bad_source = client.post(
        f"{settings.API_PREFIX}/search",
        json={"query": "ai", "limit": 10, "sources": ["invalid_source"]},
    )
    assert r_bad_source.status_code == 400
    assert "Unsupported source" in r_bad_source.json()["detail"]
