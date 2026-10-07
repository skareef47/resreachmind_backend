from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class SourcePaper(BaseModel):
    """Provider-neutral intermediate scholarly paper representation."""

    source: str = Field(..., description="Scholarly source identifier (e.g. 'semantic_scholar', 'openalex')")
    source_id: str = Field(..., description="Unique paper identifier assigned by the provider")
    title: str = Field(..., description="Paper title")
    abstract: Optional[str] = Field(default=None, description="Abstract text if available")
    authors: List[str] = Field(default_factory=list, description="List of author names")
    year: Optional[int] = Field(default=None, description="Year of publication")
    publication_date: Optional[str] = Field(default=None, description="Publication date (ISO format or YYYY-MM-DD)")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier")
    url: Optional[str] = Field(default=None, description="Direct URL or landing page for paper")
    citation_count: Optional[int] = Field(default=0, description="Total citation count reported by provider")
    venue: Optional[str] = Field(default=None, description="Journal or conference venue")
    raw_metadata: Dict[str, Any] = Field(default_factory=dict, description="Raw provider response payload")

    @field_validator("authors", mode="before")
    @classmethod
    def filter_authors(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        cleaned: List[str] = []
        for item in v:
            if isinstance(item, str) and item.strip():
                cleaned.append(item.strip())
            elif isinstance(item, dict):
                name = item.get("name") or item.get("display_name")
                if isinstance(name, str) and name.strip():
                    cleaned.append(name.strip())
        return cleaned

    @field_validator("year", mode="before")
    @classmethod
    def coerce_year(cls, v: Any) -> Optional[int]:
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
        return None

    @field_validator("raw_metadata", mode="before")
    @classmethod
    def coerce_metadata(cls, v: Any) -> Dict[str, Any]:
        return v if isinstance(v, dict) else {}



class SearchRequest(BaseModel):
    """Request model for POST /api/v1/search."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query string",
        examples=["transformer attention mechanisms"],
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Bounded maximum number of results to fetch per provider (1-100)",
        examples=[10],
    )
    sources: Optional[List[str]] = Field(
        default=None,
        description="Optional list of source providers to query (e.g. ['semantic_scholar', 'openalex']). Defaults to all.",
        examples=[["semantic_scholar", "openalex"]],
    )
    preferences: Optional[Any] = Field(
        default=None, description="Optional multi-dimensional user preferences (Phase 6)"
    )
    feedback: Optional[List[Any]] = Field(
        default=None, description="Optional user feedback items (Phase 6)"
    )
    constraints: Optional[Any] = Field(
        default=None, description="Optional hard and soft constraints (Phase 6)"
    )
    weights: Optional[Any] = Field(
        default=None, description="Optional custom utility function weights (Phase 6)"
    )
    optimize: bool = Field(
        default=False, description="Whether to execute Phase 7 global ranking optimization"
    )
    optimization_objective: Optional[str] = Field(
        default="linear_positional",
        description="Optimization objective ('linear_positional' or 'pairwise_diversity')",
    )
    diversity_lambda: float = Field(
        default=0.2, ge=0.0, description="Redundancy penalty multiplier for pairwise diversity"
    )
    include_intelligence: bool = Field(
        default=False, description="Whether to generate Phase 8 Research Intelligence report"
    )

    @field_validator("query")
    @classmethod
    def clean_query(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Query string must not be empty or whitespace only.")
        return trimmed


from app.schemas.paper import ResearchDocument
from app.schemas.retrieval import RetrievedCandidate


class SearchResponse(BaseModel):
    """Response model for POST /api/v1/search returning canonical ResearchDocument records."""

    query: str = Field(..., description="Echoed search query")
    total_results: int = Field(..., description="Total canonical research document count returned")
    provider_status: Dict[str, str] = Field(
        ...,
        description="Execution status per provider ('ok' or failure summary)",
        examples=[{"semantic_scholar": "ok", "openalex": "ok"}],
    )
    cached: Dict[str, bool] = Field(
        default_factory=dict,
        description="Cache hit indicators per queried provider",
    )
    results: List[ResearchDocument] = Field(..., description="Canonical research documents retrieved")
    candidates: List[RetrievedCandidate] = Field(
        default_factory=list,
        description="Bounded hybrid retrieval candidates with BM25, SBERT, and RRF metadata",
    )
    run_id: Optional[str] = Field(
        default=None,
        description="Persisted utility evaluation run identifier if utility model was applied",
    )
    optimization_result: Optional[Any] = Field(
        default=None,
        description="Phase 7 formal global ranking optimization result and verification report",
    )
    intelligence: Optional[Any] = Field(
        default=None,
        description="Phase 8 research intelligence report (explainability, buildability, graph, roadmap)",
    )



