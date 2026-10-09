"""
Research Architecture runtime API endpoint for ResearchMind AI.

Provides versioned endpoints:
- POST /api/v1/architecture
- GET /api/v1/architecture?paper_ids=...&query=...

Converts selected research documents and their verified implementation evidence
into a concrete, grounded technical architecture graph.
"""

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repository import DatabaseRepository
from app.db.session import get_db_session
from app.ingestion.cache import InMemoryCache
from app.intelligence.architecture import ArchitectureGenerator
from app.intelligence.models import ResearchArchitectureResponse
from app.observability.metrics import MetricsCollector
from app.retrieval.pipeline import HybridRetrievalPipeline
from app.schemas.paper import ResearchDocument
from app.schemas.source import SearchRequest
from app.sources.base import BaseSourceAdapter
from app.api.search import (
    get_cache,
    get_metrics_collector,
    get_retrieval_pipeline,
    get_search_adapters,
    search_papers,
)

logger = logging.getLogger("researchmind.api.architecture")

router = APIRouter(tags=["Architecture"])


class ArchitectureRequestPayload(BaseModel):
    """Request payload for POST /api/v1/architecture."""

    paper_ids: Optional[List[str]] = Field(
        default_factory=list, description="IDs of selected research papers to synthesize architecture from"
    )
    documents: Optional[List[ResearchDocument]] = Field(
        default=None, description="Explicit research documents (optional override)"
    )
    query: Optional[str] = Field(
        default=None, description="Research query to retrieve papers for if paper_ids not in cache"
    )
    selection_mode: Optional[str] = Field(
        default="optimized", description="Selection mode: optimized or manual"
    )
    constraints: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Deployment, latency, or compute constraints"
    )


@router.post(
    "/architecture",
    response_model=ResearchArchitectureResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate research-grounded technical architecture",
    description="Synthesizes a concrete technical implementation architecture from selected research papers.",
)
async def generate_architecture_endpoint(
    payload: ArchitectureRequestPayload,
    session: AsyncSession = Depends(get_db_session),
    adapters: Dict[str, BaseSourceAdapter] = Depends(get_search_adapters),
    cache: InMemoryCache = Depends(get_cache),
    pipeline: HybridRetrievalPipeline = Depends(get_retrieval_pipeline),
    metrics: MetricsCollector = Depends(get_metrics_collector),
) -> ResearchArchitectureResponse:
    """Generate architecture from selected papers, explicit documents, or query."""
    docs: List[ResearchDocument] = []
    repo = DatabaseRepository(session=session)

    # 1. Use explicit documents if provided in payload
    if payload.documents and len(payload.documents) > 0:
        docs.extend(payload.documents)

    # 2. Look up papers by ID in database repository
    if payload.paper_ids and len(payload.paper_ids) > 0:
        for pid in payload.paper_ids:
            found = await repo.get_document(pid)
            if found and found.internal_id not in [d.internal_id for d in docs]:
                docs.append(found)

    # 3. If no documents found yet but a query is provided, execute search
    if not docs and payload.query and payload.query.strip():
        logger.info("Retrieving candidate papers for query '%s' to build architecture", payload.query)
        search_req = SearchRequest(
            query=payload.query.strip(),
            limit=5,
            optimize=True,
        )
        search_res = await search_papers(
            request=search_req,
            session=session,
            adapters=adapters,
            cache=cache,
            pipeline=pipeline,
            metrics=metrics,
        )
        if search_res.results:
            docs.extend(search_res.results)

    # 4. If still empty, raise 400 Bad Request
    if not docs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid research papers found. Please provide valid 'paper_ids', 'documents', or a 'query'.",
        )

    generator = ArchitectureGenerator()
    try:
        arch_response = generator.generate_architecture(
            documents=docs,
            constraints=payload.constraints,
            query=payload.query,
            selection_mode=payload.selection_mode,
        )
        return arch_response
    except Exception as exc:
        logger.exception("Architecture generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to synthesize architecture: {str(exc)}",
        )


@router.get(
    "/architecture",
    response_model=ResearchArchitectureResponse,
    status_code=status.HTTP_200_OK,
    summary="Query research-grounded technical architecture (GET)",
    description="Query-based architecture generation endpoint.",
)
async def get_architecture_endpoint(
    query: Optional[str] = Query(default=None, description="Research query topic"),
    paper_ids: Optional[List[str]] = Query(default=None, description="Target paper internal IDs"),
    selection_mode: Optional[str] = Query(default="optimized", description="Selection mode: optimized or manual"),
    target_environment: Optional[str] = Query(default="cloud", description="cloud, edge, or local"),
    max_gpu_memory_gb: Optional[int] = Query(default=None, description="VRAM budget in GB"),
    latency_budget_ms: Optional[int] = Query(default=None, description="Latency budget in ms"),
    session: AsyncSession = Depends(get_db_session),
    adapters: Dict[str, BaseSourceAdapter] = Depends(get_search_adapters),
    cache: InMemoryCache = Depends(get_cache),
    pipeline: HybridRetrievalPipeline = Depends(get_retrieval_pipeline),
    metrics: MetricsCollector = Depends(get_metrics_collector),
) -> ResearchArchitectureResponse:
    """Convenience GET endpoint for architecture synthesis."""
    constraints: Dict[str, Any] = {"target_environment": target_environment}
    if max_gpu_memory_gb is not None:
        constraints["max_gpu_memory_gb"] = max_gpu_memory_gb
    if latency_budget_ms is not None:
        constraints["latency_budget_ms"] = latency_budget_ms

    payload = ArchitectureRequestPayload(
        paper_ids=paper_ids or [],
        query=query,
        selection_mode=selection_mode,
        constraints=constraints,
    )
    return await generate_architecture_endpoint(
        payload=payload,
        session=session,
        adapters=adapters,
        cache=cache,
        pipeline=pipeline,
        metrics=metrics,
    )
