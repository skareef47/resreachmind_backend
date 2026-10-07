from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProvenanceRecord(BaseModel):
    """Provenance entry tracing scholarly source origin and retrieval history."""

    source: str = Field(..., description="Origin provider name (e.g. 'semantic_scholar', 'openalex')")
    source_id: str = Field(..., description="Provider's native record identifier")
    retrieved_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of retrieval",
    )
    original_url: Optional[str] = Field(default=None, description="Original URL or landing page from provider")
    raw_metadata: Dict[str, Any] = Field(default_factory=dict, description="Raw provider metadata payload")


class ResearchDocument(BaseModel):
    """Canonical representation for all scholarly papers across sources."""

    internal_id: str = Field(..., description="Deterministic global unique identifier")
    source: str = Field(..., description="Primary or consolidated provider name(s)")
    source_id: str = Field(..., description="Primary provider record ID")
    title: str = Field(..., description="Normalized paper title")
    abstract: Optional[str] = Field(default=None, description="Normalized abstract text")
    authors: List[str] = Field(default_factory=list, description="Ordered list of author names")
    publication_year: Optional[int] = Field(default=None, description="Year of publication")
    publication_date: Optional[str] = Field(default=None, description="ISO publication date (YYYY-MM-DD)")
    doi: Optional[str] = Field(default=None, description="Normalized clean DOI (e.g. '10.1000/182')")
    url: Optional[str] = Field(default=None, description="Canonical or landing page URL")
    citation_count: Optional[int] = Field(default=0, description="Total citations reported")
    references: List[str] = Field(default_factory=list, description="Referenced paper IDs or citation keys")
    citations: List[str] = Field(default_factory=list, description="Citing paper IDs or citation keys")
    venue: Optional[str] = Field(default=None, description="Journal or conference publication venue")
    fields_of_study: List[str] = Field(default_factory=list, description="High-level fields of study")
    concepts: List[str] = Field(default_factory=list, description="Topic concepts and ontology tags")
    datasets: List[str] = Field(default_factory=list, description="Identified datasets referenced")
    methods: List[str] = Field(default_factory=list, description="Identified algorithms or methodologies")
    models: List[str] = Field(default_factory=list, description="Identified ML/statistical models")
    code_url: Optional[str] = Field(default=None, description="Repository URL (e.g. GitHub/GitLab)")
    open_access: Optional[bool] = Field(default=None, description="Open access accessibility status")
    license: Optional[str] = Field(default=None, description="Publication license (e.g. 'cc-by')")
    provenance: List[ProvenanceRecord] = Field(
        default_factory=list,
        description="Audit trail of all merged source origins",
    )
    raw_source_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Consolidated raw source metadata",
    )
    retrieval_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of initial retrieval",
    )
