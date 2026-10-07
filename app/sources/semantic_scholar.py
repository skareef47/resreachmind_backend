import logging
from typing import Any, Dict, List, Optional

import httpx

from app.schemas.source import SourcePaper
from app.sources.base import BaseSourceAdapter

logger = logging.getLogger("researchmind.sources.semantic_scholar")

SEMANTIC_SCHOLAR_FIELDS = (
    "paperId,title,abstract,authors,year,publicationDate,externalIds,url,citationCount,venue"
)


class SemanticScholarAdapter(BaseSourceAdapter):
    """Adapter for Semantic Scholar Academic Graph API."""

    source_name: str = "semantic_scholar"

    def __init__(
        self,
        base_url: str = "https://api.semanticscholar.org/graph/v1",
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            client=client,
        )

    def _build_headers(self) -> Dict[str, str]:
        """Build headers with optional backend API key."""
        headers: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "ResearchMindAI/1.0",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _parse_paper(self, item: Dict[str, Any]) -> Optional[SourcePaper]:
        """Map Semantic Scholar paper payload to SourcePaper schema."""
        paper_id = item.get("paperId")
        title = item.get("title")
        if not paper_id or not title:
            return None

        # Authors
        authors: List[str] = []
        raw_authors = item.get("authors") or []
        for a in raw_authors:
            if isinstance(a, dict) and a.get("name"):
                authors.append(a["name"])

        # DOI
        external_ids = item.get("externalIds") or {}
        doi = external_ids.get("DOI") if isinstance(external_ids, dict) else None

        # URL
        url = item.get("url") or (f"https://www.semanticscholar.org/paper/{paper_id}")

        return SourcePaper(
            source=self.source_name,
            source_id=str(paper_id),
            title=title.strip(),
            abstract=item.get("abstract"),
            authors=authors,
            year=item.get("year"),
            publication_date=item.get("publicationDate"),
            doi=doi,
            url=url,
            citation_count=item.get("citationCount") or 0,
            venue=item.get("venue"),
            raw_metadata=item,
        )

    async def search(self, query: str, limit: int = 10) -> List[SourcePaper]:
        """Query Semantic Scholar search endpoint and return normalized SourcePaper records."""
        url = f"{self.base_url}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": SEMANTIC_SCHOLAR_FIELDS,
        }
        headers = self._build_headers()

        payload = await self._request_with_retry(
            method="GET",
            url=url,
            params=params,
            headers=headers,
        )

        raw_papers = payload.get("data") or []
        results: List[SourcePaper] = []
        for item in raw_papers:
            if isinstance(item, dict):
                paper = self._parse_paper(item)
                if paper:
                    results.append(paper)

        return results
