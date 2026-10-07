import asyncio
from typing import Any, Dict
import httpx
import pytest

from app.sources.base import (
    SourceError,
    SourceHttpError,
    SourceMalformedResponseError,
    SourceRateLimitError,
    SourceTimeoutError,
    sanitize_headers,
)
from app.sources.openalex import OpenAlexAdapter, reconstruct_abstract
from app.sources.semantic_scholar import SemanticScholarAdapter


def test_sanitize_headers() -> None:
    """Verify secrets are redacted from headers."""
    raw = {
        "x-api-key": "secret-s2-key-12345",
        "Authorization": "Bearer secret-token-abc",
        "Accept": "application/json",
        "User-Agent": "ResearchMind/1.0",
    }
    sanitized = sanitize_headers(raw)
    assert sanitized["x-api-key"] == "[REDACTED]"
    assert sanitized["Authorization"] == "[REDACTED]"
    assert sanitized["Accept"] == "application/json"
    assert sanitized["User-Agent"] == "ResearchMind/1.0"


def test_reconstruct_openalex_abstract() -> None:
    """Verify OpenAlex inverted index abstract reconstruction."""
    inverted_index = {
        "Attention": [0],
        "is": [1],
        "all": [2],
        "you": [3],
        "need.": [4],
    }
    result = reconstruct_abstract(inverted_index)
    assert result == "Attention is all you need."

    assert reconstruct_abstract(None) is None
    assert reconstruct_abstract({}) is None


def test_semantic_scholar_success() -> None:
    """Verify Semantic Scholar search adapter parses response correctly."""
    async def _run() -> None:
        mock_payload = {
            "total": 1,
            "offset": 0,
            "data": [
                {
                    "paperId": "s2_abc123",
                    "title": "Deep Learning for Natural Language Processing",
                    "abstract": "An overview of transformer models.",
                    "authors": [{"name": "Alice Smith"}, {"name": "Bob Jones"}],
                    "year": 2023,
                    "publicationDate": "2023-05-15",
                    "externalIds": {"DOI": "10.1000/182"},
                    "url": "https://www.semanticscholar.org/paper/s2_abc123",
                    "citationCount": 42,
                    "venue": "ACL 2023",
                }
            ],
        }

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path.endswith("/paper/search")
            assert "query=transformer" in request.url.query.decode()
            assert request.headers.get("x-api-key") == "test-key"
            return httpx.Response(200, json=mock_payload)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            adapter = SemanticScholarAdapter(
                base_url="https://api.semanticscholar.org/graph/v1",
                api_key="test-key",
                client=client,
            )
            papers = await adapter.search(query="transformer", limit=5)

        assert len(papers) == 1
        p = papers[0]
        assert p.source == "semantic_scholar"
        assert p.source_id == "s2_abc123"
        assert p.title == "Deep Learning for Natural Language Processing"
        assert p.abstract == "An overview of transformer models."
        assert p.authors == ["Alice Smith", "Bob Jones"]
        assert p.year == 2023
        assert p.doi == "10.1000/182"
        assert p.citation_count == 42
        assert p.venue == "ACL 2023"

    asyncio.run(_run())


def test_openalex_success() -> None:
    """Verify OpenAlex search adapter parses works and inverted index abstract."""
    async def _run() -> None:
        mock_payload = {
            "meta": {"count": 1},
            "results": [
                {
                    "id": "https://openalex.org/W987654321",
                    "display_name": "Attention Is All You Need",
                    "publication_year": 2017,
                    "publication_date": "2017-06-12",
                    "doi": "https://doi.org/10.48550/arxiv.1706.03762",
                    "cited_by_count": 85000,
                    "abstract_inverted_index": {
                        "The": [0],
                        "dominant": [1],
                        "sequence": [2],
                        "transduction": [3],
                        "models": [4],
                    },
                    "authorships": [
                        {"author": {"display_name": "Ashish Vaswani"}},
                        {"author": {"display_name": "Noam Shazeer"}},
                    ],
                    "primary_location": {
                        "source": {"display_name": "NeurIPS"},
                        "landing_page_url": "https://arxiv.org/abs/1706.03762",
                    },
                }
            ],
        }

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/works"
            assert "search=attention" in request.url.query.decode()
            return httpx.Response(200, json=mock_payload)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            adapter = OpenAlexAdapter(
                base_url="https://api.openalex.org",
                mailto="researcher@example.com",
                client=client,
            )
            papers = await adapter.search(query="attention", limit=5)

        assert len(papers) == 1
        p = papers[0]
        assert p.source == "openalex"
        assert p.source_id == "W987654321"
        assert p.title == "Attention Is All You Need"
        assert p.abstract == "The dominant sequence transduction models"
        assert p.authors == ["Ashish Vaswani", "Noam Shazeer"]
        assert p.year == 2017
        assert p.doi == "10.48550/arxiv.1706.03762"
        assert p.citation_count == 85000
        assert p.venue == "NeurIPS"

    asyncio.run(_run())


def test_adapter_timeout_raises_source_timeout_error() -> None:
    """Verify timeout exception triggers SourceTimeoutError."""
    async def _run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Connection timed out", request=request)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            adapter = SemanticScholarAdapter(
                base_url="https://api.semanticscholar.org/graph/v1",
                timeout=1.0,
                max_retries=1,
                backoff_factor=0.01,
                client=client,
            )
            with pytest.raises(SourceTimeoutError) as exc_info:
                await adapter.search(query="neural networks")
            assert "semantic_scholar" in str(exc_info.value)

    asyncio.run(_run())


def test_adapter_429_retry_and_success() -> None:
    """Verify 429 Too Many Requests retries with backoff and succeeds on subsequent try."""
    async def _run() -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(429, headers={"Retry-After": "0.01"})
            return httpx.Response(200, json={"data": [{"paperId": "p1", "title": "Paper One"}]})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            adapter = SemanticScholarAdapter(
                base_url="https://api.semanticscholar.org/graph/v1",
                max_retries=2,
                backoff_factor=0.01,
                client=client,
            )
            papers = await adapter.search(query="ai")

        assert calls == 2
        assert len(papers) == 1
        assert papers[0].title == "Paper One"

    asyncio.run(_run())


def test_adapter_429_exhausted_raises_rate_limit_error() -> None:
    """Verify 429 on all attempts raises SourceRateLimitError."""
    async def _run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"Retry-After": "0.01"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            adapter = SemanticScholarAdapter(
                base_url="https://api.semanticscholar.org/graph/v1",
                max_retries=1,
                backoff_factor=0.01,
                client=client,
            )
            with pytest.raises(SourceRateLimitError) as exc_info:
                await adapter.search(query="ai")
            assert "Rate limit exceeded" in str(exc_info.value)

    asyncio.run(_run())


def test_adapter_server_error_and_malformed_response() -> None:
    """Verify 500 error and malformed JSON payloads raise proper structured errors."""
    async def _run() -> None:
        def error_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        transport = httpx.MockTransport(error_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            adapter = OpenAlexAdapter(
                base_url="https://api.openalex.org",
                max_retries=0,
                client=client,
            )
            with pytest.raises(SourceHttpError) as exc_info:
                await adapter.search(query="test")
            assert exc_info.value.status_code == 500

        def bad_json_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"invalid { json")

        bad_transport = httpx.MockTransport(bad_json_handler)
        async with httpx.AsyncClient(transport=bad_transport) as client:
            adapter2 = OpenAlexAdapter(
                base_url="https://api.openalex.org",
                max_retries=0,
                client=client,
            )
            with pytest.raises(SourceMalformedResponseError):
                await adapter2.search(query="test")

    asyncio.run(_run())
