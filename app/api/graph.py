"""
Knowledge Graph runtime API endpoint for ResearchMind AI.

Provides versioned endpoints:
- GET /api/v1/knowledge-graph?query=...&limit=...
- POST /api/v1/knowledge-graph (payload with query and/or documents)

Synthesizes interconnected multi-entity knowledge graphs from retrieved research candidates
(Papers, Authors, Institutions, Venues, Concepts, Models, Datasets, Methods, Tasks, Technologies, Repositories).
"""

import logging
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.ingestion.cache import InMemoryCache
from app.intelligence.graph import KnowledgeGraphBuilder
from app.intelligence.models import KnowledgeGraph
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

logger = logging.getLogger("researchmind.api.graph")

router = APIRouter(tags=["Knowledge Graph"])


class KnowledgeGraphPayload(BaseModel):
    """Request payload for POST /api/v1/knowledge-graph."""

    query: Optional[str] = Field(default=None, description="Research query to retrieve papers for")
    documents: Optional[List[ResearchDocument]] = Field(
        default=None, description="Explicit papers to construct knowledge graph from"
    )
    limit: int = Field(default=10, ge=1, le=50, description="Maximum documents to retrieve if query provided")
    sources: Optional[List[str]] = Field(
        default=None, description="Scholarly provider names (e.g. ['openalex', 'semantic_scholar'])"
    )


@router.get(
    "/knowledge-graph",
    response_model=KnowledgeGraph,
    summary="Retrieve Scholarly Knowledge Graph (GET)",
    description="Retrieve live, evidence-grounded knowledge graph synthesized from retrieved research papers.",
)
async def get_knowledge_graph(
    query: str = Query(..., description="Research query or topic"),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum papers to retrieve"),
    adapters: Dict[str, BaseSourceAdapter] = Depends(get_search_adapters),
    cache: InMemoryCache = Depends(get_cache),
    pipeline: HybridRetrievalPipeline = Depends(get_retrieval_pipeline),
    metrics: MetricsCollector = Depends(get_metrics_collector),
    session: AsyncSession = Depends(get_db_session),
) -> KnowledgeGraph:
    try:
        search_req = SearchRequest(
            query=query,
            limit=limit,
            optimize=True,
        )
        search_res = await search_papers(
            request=search_req,
            adapters=adapters,
            cache=cache,
            pipeline=pipeline,
            metrics=metrics,
            session=session,
        )
        if search_res.knowledge_graph is not None:
            return search_res.knowledge_graph

        docs = search_res.results or search_res.papers or []
        builder = KnowledgeGraphBuilder()
        return builder.build_graph(docs)
    except Exception as exc:
        logger.exception("Failed to build knowledge graph for query '%s': %s", query, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge graph synthesis failed: {str(exc)}",
        )


@router.post(
    "/knowledge-graph",
    response_model=KnowledgeGraph,
    summary="Construct Scholarly Knowledge Graph (POST)",
    description="Construct multi-entity knowledge graph from explicit document list or by querying scholarly providers.",
)
async def post_knowledge_graph(
    payload: KnowledgeGraphPayload,
    adapters: Dict[str, BaseSourceAdapter] = Depends(get_search_adapters),
    cache: InMemoryCache = Depends(get_cache),
    pipeline: HybridRetrievalPipeline = Depends(get_retrieval_pipeline),
    metrics: MetricsCollector = Depends(get_metrics_collector),
    session: AsyncSession = Depends(get_db_session),
) -> KnowledgeGraph:
    try:
        builder = KnowledgeGraphBuilder()
        if payload.documents and len(payload.documents) > 0:
            return builder.build_graph(payload.documents)

        if not payload.query or not payload.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'documents' list or 'query' string must be provided.",
            )

        search_req = SearchRequest(
            query=payload.query,
            limit=payload.limit,
            sources=payload.sources,
            optimize=True,
        )
        search_res = await search_papers(
            request=search_req,
            adapters=adapters,
            cache=cache,
            pipeline=pipeline,
            metrics=metrics,
            session=session,
        )
        if search_res.knowledge_graph is not None:
            return search_res.knowledge_graph

        docs = search_res.results or search_res.papers or []
        return builder.build_graph(docs)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to construct knowledge graph: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge graph synthesis failed: {str(exc)}",
        )
