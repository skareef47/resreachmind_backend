import logging
from typing import Any, Dict, List, Optional

import httpx

from app.schemas.source import SourcePaper
from app.sources.base import BaseSourceAdapter

logger = logging.getLogger("researchmind.sources.openalex")


def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """Reconstruct human-readable abstract from OpenAlex abstract_inverted_index."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return None

    try:
        word_positions: List[tuple[int, str]] = []
        for word, positions in inverted_index.items():
            if isinstance(positions, list):
                for pos in positions:
                    if isinstance(pos, int):
                        word_positions.append((pos, word))

        if not word_positions:
            return None

        word_positions.sort(key=lambda x: x[0])
        return " ".join(word for _, word in word_positions)
    except Exception as e:
        logger.debug("Failed to reconstruct abstract from inverted index: %s", e)
        return None


class OpenAlexAdapter(BaseSourceAdapter):
    """Adapter for OpenAlex Works API."""

    source_name: str = "openalex"

    def __init__(
        self,
        base_url: str = "https://api.openalex.org",
        api_key: Optional[str] = None,
        mailto: Optional[str] = None,
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
        self.mailto = mailto

    def _build_headers(self) -> Dict[str, str]:
        """Build headers with optional polite pool mailto or backend API key."""
        headers: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": f"ResearchMindAI/1.0 (mailto:{self.mailto})" if self.mailto else "ResearchMindAI/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _parse_work(self, item: Dict[str, Any]) -> Optional[SourcePaper]:
        """Map OpenAlex work payload to SourcePaper schema."""
        work_id = item.get("id")
        title = item.get("display_name") or item.get("title")
        if not work_id or not title:
            return None

        # Clean work ID (e.g., https://openalex.org/W123456 -> W123456 or full URI)
        short_id = work_id.replace("https://openalex.org/", "") if isinstance(work_id, str) else str(work_id)

        # Authors
        authors: List[str] = []
        raw_authorships = item.get("authorships") or []
        for auth in raw_authorships:
            if isinstance(auth, dict):
                author_obj = auth.get("author") or {}
                if isinstance(author_obj, dict) and author_obj.get("display_name"):
                    authors.append(author_obj["display_name"])

        # DOI
        doi_raw = item.get("doi")
        doi = doi_raw.replace("https://doi.org/", "") if isinstance(doi_raw, str) else None

        # Abstract
        abstract = reconstruct_abstract(item.get("abstract_inverted_index"))

        # Venue
        venue = None
        primary_location = item.get("primary_location") or {}
        if isinstance(primary_location, dict):
            source_info = primary_location.get("source") or {}
            if isinstance(source_info, dict):
                venue = source_info.get("display_name")

        # URL
        url = (
            item.get("doi")
            or (primary_location.get("landing_page_url") if isinstance(primary_location, dict) else None)
            or work_id
        )

        return SourcePaper(
            source=self.source_name,
            source_id=short_id,
            title=title.strip(),
            abstract=abstract,
            authors=authors,
            year=item.get("publication_year"),
            publication_date=item.get("publication_date"),
            doi=doi,
            url=url,
            citation_count=item.get("cited_by_count") or 0,
            venue=venue,
            raw_metadata=item,
        )

    async def search(self, query: str, limit: int = 10) -> List[SourcePaper]:
        """Query OpenAlex works endpoint and return normalized SourcePaper records."""
        url = f"{self.base_url}/works"
        params: Dict[str, Any] = {
            "search": query,
            "per-page": limit,
        }
        if self.mailto:
            params["mailto"] = self.mailto

        headers = self._build_headers()

        payload = await self._request_with_retry(
            method="GET",
            url=url,
            params=params,
            headers=headers,
        )

        raw_results = payload.get("results") or []
        results: List[SourcePaper] = []
        for item in raw_results:
            if isinstance(item, dict):
                paper = self._parse_work(item)
                if paper:
                    results.append(paper)

        return results
