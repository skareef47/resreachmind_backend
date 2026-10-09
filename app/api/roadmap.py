"""
Research Implementation Roadmap runtime API endpoint for ResearchMind AI.

Provides versioned endpoints:
- POST /api/v1/roadmap
- GET /api/v1/roadmap?paper_ids=...&query=...

Converts selected research documents, extracted implementation evidence,
generated architecture, and user constraints into an actionable 8-phase implementation roadmap.
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
from app.intelligence.models import ResearchRoadmapResponse
from app.intelligence.roadmap import RoadmapGenerator
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

logger = logging.getLogger("researchmind.api.roadmap")

router = APIRouter(tags=["Roadmap"])


class RoadmapRequestPayload(BaseModel):
    """Request payload for POST /api/v1/roadmap."""

    paper_ids: Optional[List[str]] = Field(
        default_factory=list, description="IDs of selected research papers"
    )
    architecture: Optional[Dict[str, Any]] = Field(
        default=None, description="Pre-generated technical architecture specification"
    )
    constraints: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="User constraints (limited GPU, limited time, public datasets, etc.)"
    )
    selection_mode: Optional[str] = Field(
        default="optimized", description="Selection mode: optimized or manual"
    )
    documents: Optional[List[ResearchDocument]] = Field(
        default=None, description="Explicit research documents (optional override)"
    )
    query: Optional[str] = Field(
        default=None, description="Research query to retrieve papers for if paper_ids not in cache"
    )


@router.post(
    "/roadmap",
    response_model=ResearchRoadmapResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Actionable Implementation Roadmap",
    description="Synthesize an actionable 8-phase implementation roadmap from selected research papers, architecture, and constraints.",
)
async def generate_roadmap_endpoint(
    payload: RoadmapRequestPayload,
    session: AsyncSession = Depends(get_db_session),
    adapters: Dict[str, BaseSourceAdapter] = Depends(get_search_adapters),
    cache: InMemoryCache = Depends(get_cache),
    pipeline: HybridRetrievalPipeline = Depends(get_retrieval_pipeline),
    metrics: MetricsCollector = Depends(get_metrics_collector),
) -> ResearchRoadmapResponse:
    """Generate roadmap from selected papers, architecture, and constraints."""
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
        logger.info("Retrieving candidate papers for query '%s' to build roadmap", payload.query)
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
        if search_res and search_res.results:
            docs.extend(search_res.results)

    # If still no documents, return empty roadmap with message
    if not docs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid papers found for roadmap generation. Please supply valid 'paper_ids', 'documents', or a 'query'.",
        )

    # 4. Validate or synthesize technical architecture
    arch_data = payload.architecture
    arch_gen = ArchitectureGenerator()
    should_regenerate = False

    if not arch_data:
        should_regenerate = True
    elif isinstance(arch_data, dict):
        arch_pids = arch_data.get("selected_paper_ids") or arch_data.get("papers_used") or []
        doc_ids = [d.internal_id for d in docs if d.internal_id]
        if doc_ids and arch_pids and not any(pid in doc_ids for pid in arch_pids):
            logger.info("Stale architecture detected in roadmap payload (paper ID mismatch). Regenerating fresh architecture.")
            should_regenerate = True

    if should_regenerate:
        arch_obj = arch_gen.generate_architecture(
            documents=docs,
            constraints=payload.constraints,
            query=payload.query,
            selection_mode=payload.selection_mode,
        )
        arch_data = arch_obj.model_dump()

    logger.info(
        "ROADMAP API CALL: query=%s, doc_count=%d, paper_ids=%s",
        payload.query,
        len(docs),
        [d.internal_id for d in docs],
    )

    # 5. Generate roadmap
    roadmap_gen = RoadmapGenerator()
    roadmap_res = roadmap_gen.generate_roadmap(
        documents=docs,
        architecture=arch_data,
        constraints=payload.constraints,
        query=payload.query,
        selection_mode=payload.selection_mode,
    )

    # 6. Save roadmap in database
    try:
        await repo.save_roadmap(roadmap_res)
    except Exception as exc:
        logger.warning("Could not persist roadmap to DB (proceeding): %s", exc)

    return roadmap_res


@router.get(
    "/roadmap",
    response_model=ResearchRoadmapResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve or generate research implementation roadmap",
    description="Retrieve research roadmap by query and optional constraints.",
)
async def get_roadmap_endpoint(
    query: str = Query(..., description="Research query topic to generate roadmap for"),
    limited_gpu: Optional[bool] = Query(default=False, description="Enable limited GPU mode (<=8GB VRAM)"),
    limited_time: Optional[bool] = Query(default=False, description="Enable fast-track fine-tuning mode (<1 week)"),
    public_datasets_only: Optional[bool] = Query(default=False, description="Restrict strictly to public datasets"),
    open_source_only: Optional[bool] = Query(default=False, description="Restrict strictly to open-source licenses"),
    session: AsyncSession = Depends(get_db_session),
    adapters: Dict[str, BaseSourceAdapter] = Depends(get_search_adapters),
    cache: InMemoryCache = Depends(get_cache),
    pipeline: HybridRetrievalPipeline = Depends(get_retrieval_pipeline),
    metrics: MetricsCollector = Depends(get_metrics_collector),
) -> ResearchRoadmapResponse:
    """GET endpoint to synthesize an implementation roadmap for a search query."""
    constraints: Dict[str, Any] = {
        "limited_gpu": limited_gpu,
        "limited_time": limited_time,
        "public_datasets_only": public_datasets_only,
        "open_source_only": open_source_only,
    }
    if limited_gpu:
        constraints["max_gpu_memory_gb"] = 8

    req_payload = RoadmapRequestPayload(
        query=query,
        constraints=constraints,
    )
    return await generate_roadmap_endpoint(
        payload=req_payload,
        session=session,
        adapters=adapters,
        cache=cache,
        pipeline=pipeline,
        metrics=metrics,
    )
