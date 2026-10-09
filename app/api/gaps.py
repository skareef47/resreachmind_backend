"""
Research Gap Intelligence runtime API endpoint for ResearchMind AI.

Provides versioned endpoints:
- POST /api/v1/research-gaps
- GET /api/v1/research-gaps?query=...

Analyzes the research landscape across retrieved candidates to identify evidence-backed
potential research gaps and actionable research opportunities.
"""

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repository import DatabaseRepository
from app.db.session import get_db_session
from app.ingestion.cache import InMemoryCache
from app.intelligence.gaps import ResearchGapAnalyzer
from app.intelligence.models import ResearchGapsResponse
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

logger = logging.getLogger("researchmind.api.gaps")

router = APIRouter(tags=["Research Gaps"])


class ResearchGapsRequestPayload(BaseModel):
    """Request payload for POST /api/v1/research-gaps."""

    query: Optional[str] = Field(
        default=None, description="Search query or domain focus"
    )
    paper_ids: Optional[List[str]] = Field(
        default_factory=list, description="IDs of candidate research papers to analyze"
    )
    documents: Optional[List[ResearchDocument]] = Field(
        default=None, description="Explicit research documents (optional in-memory override)"
    )


async def _resolve_candidate_documents(
    paper_ids: Optional[List[str]],
    documents: Optional[List[ResearchDocument]],
    query: Optional[str],
    cache: InMemoryCache,
    db: AsyncSession,
    retrieval_pipeline: HybridRetrievalPipeline,
    adapters: Dict[str, BaseSourceAdapter],
    metrics_collector: MetricsCollector,
) -> List[ResearchDocument]:
    """Resolve documents from payload, cache, database, or live retrieval."""
    # 1. Use explicit documents if passed
    if documents and len(documents) > 0:
        return documents

    target_docs: List[ResearchDocument] = []
    repo = DatabaseRepository(db)

    # 2. Try resolving requested paper_ids from cache
    if paper_ids and len(paper_ids) > 0:
        for pid in paper_ids:
            cached_doc = cache.get(f"doc_{pid}") or cache.get(pid)
            if cached_doc:
                try:
                    target_docs.append(ResearchDocument.model_validate(cached_doc))
                except Exception:
                    pass

        # 3. If missing from cache, query database repository
        if len(target_docs) < len(paper_ids):
            db_papers = await repo.list_documents(limit=100)
            db_docs_map = {p.internal_id: p for p in db_papers}
            for pid in paper_ids:
                if any(d.internal_id == pid or d.paper_id == pid for d in target_docs):
                    continue
                if pid in db_docs_map:
                    target_docs.append(db_docs_map[pid])

    # 4. If still empty, use query or fallback to recent papers
    if not target_docs:
        effective_query = query.strip() if query and query.strip() else None
        if effective_query:
            logger.info("Executing retrieval for gap analysis on query: '%s'", effective_query)
            try:
                search_req = SearchRequest(
                    query=effective_query,
                    limit=8,
                    sources=["semantic_scholar", "openalex"],
                    optimize=False,
                )
                search_resp = await search_papers(
                    request=search_req,
                    adapters=adapters,
                    cache=cache,
                    pipeline=retrieval_pipeline,
                    metrics=metrics_collector,
                    session=db,
                )
                target_docs = [
                    ResearchDocument.model_validate(doc_dict)
                    for doc_dict in search_resp.results
                ]
            except Exception as exc:
                logger.warning("Dynamic search failed for gap analysis: %s", exc)

        if not target_docs:
            db_papers = await repo.list_documents(limit=8)
            target_docs.extend(db_papers)

    return target_docs


@router.post(
    "/research-gaps",
    response_model=ResearchGapsResponse,
    summary="Analyze candidate papers to identify evidence-backed potential research gaps",
    status_code=status.HTTP_200_OK,
)
async def generate_research_gaps(
    payload: ResearchGapsRequestPayload,
    cache: InMemoryCache = Depends(get_cache),
    db: AsyncSession = Depends(get_db_session),
    retrieval_pipeline: HybridRetrievalPipeline = Depends(get_retrieval_pipeline),
    adapters: Dict[str, BaseSourceAdapter] = Depends(get_search_adapters),
    metrics_collector: MetricsCollector = Depends(get_metrics_collector),
) -> ResearchGapsResponse:
    """
    POST /api/v1/research-gaps

    Analyzes retrieved research papers and synthesizes:
    - Potential Research Gaps (categorized by method, dataset, evaluation, application, scalability)
    - Empirical evidence quotes and citations from actual candidate papers
    - Why the gap appears in the candidate set
    - Evidence-backed observation vs AI-generated hypothesis
    - Actionable future research directions
    - Epistemologically grounded disclaimer: "Limited evidence was found within the analyzed candidate set."
    """
    analyzer = ResearchGapAnalyzer()

    target_docs = await _resolve_candidate_documents(
        paper_ids=payload.paper_ids,
        documents=payload.documents,
        query=payload.query,
        cache=cache,
        db=db,
        retrieval_pipeline=retrieval_pipeline,
        adapters=adapters,
        metrics_collector=metrics_collector,
    )

    if not target_docs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No candidate papers found to analyze. Please provide valid paper_ids, documents, or a query.",
        )

    # Filter to specific paper_ids if requested and provided in target_docs
    if payload.paper_ids and len(payload.paper_ids) > 0:
        filtered = [d for d in target_docs if (d.internal_id in payload.paper_ids or d.paper_id in payload.paper_ids)]
        if filtered:
            target_docs = filtered

    response = analyzer.analyze_gaps(documents=target_docs, query=payload.query)
    return response


@router.get(
    "/research-gaps",
    response_model=ResearchGapsResponse,
    summary="GET endpoint for research gap intelligence",
    status_code=status.HTTP_200_OK,
)
async def get_research_gaps(
    query: Optional[str] = Query(None, description="Research query to evaluate"),
    paper_ids: Optional[List[str]] = Query(None, description="Optional paper IDs"),
    cache: InMemoryCache = Depends(get_cache),
    db: AsyncSession = Depends(get_db_session),
    retrieval_pipeline: HybridRetrievalPipeline = Depends(get_retrieval_pipeline),
    adapters: Dict[str, BaseSourceAdapter] = Depends(get_search_adapters),
    metrics_collector: MetricsCollector = Depends(get_metrics_collector),
) -> ResearchGapsResponse:
    """
    GET /api/v1/research-gaps?query=...

    Convenience GET endpoint for research gap analysis.
    """
    analyzer = ResearchGapAnalyzer()

    target_docs = await _resolve_candidate_documents(
        paper_ids=paper_ids,
        documents=None,
        query=query,
        cache=cache,
        db=db,
        retrieval_pipeline=retrieval_pipeline,
        adapters=adapters,
        metrics_collector=metrics_collector,
    )

    if not target_docs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No research papers could be retrieved to perform gap analysis.",
        )

    response = analyzer.analyze_gaps(documents=target_docs, query=query)
    return response
