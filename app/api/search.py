import asyncio
import logging
import time
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db.repository import DatabaseRepository
from app.db.session import get_db_session
from app.ingestion.cache import InMemoryCache, generate_cache_key, get_cache
from app.ingestion.deduplicate import deduplicate_documents
from app.ingestion.normalize import normalize_source_paper
from app.observability.metrics import MetricsCollector, get_metrics_collector
from app.retrieval.pipeline import HybridRetrievalPipeline, get_retrieval_pipeline
from app.schemas.paper import ResearchDocument
from app.schemas.source import SearchRequest, SearchResponse, SourcePaper
from app.sources.base import BaseSourceAdapter
from app.sources.openalex import OpenAlexAdapter
from app.sources.semantic_scholar import SemanticScholarAdapter

logger = logging.getLogger("researchmind.api.search")

router = APIRouter(tags=["Search"])


def get_search_adapters(
    settings: Settings = Depends(get_settings),
) -> Dict[str, BaseSourceAdapter]:
    """Provide configured source adapter instances."""
    return {
        "semantic_scholar": SemanticScholarAdapter(
            base_url=settings.SEMANTIC_SCHOLAR_BASE_URL,
            api_key=settings.SEMANTIC_SCHOLAR_API_KEY,
            timeout=settings.HTTP_TIMEOUT_SECONDS,
            max_retries=settings.HTTP_MAX_RETRIES,
            backoff_factor=settings.HTTP_RETRY_BACKOFF_FACTOR,
        ),
        "openalex": OpenAlexAdapter(
            base_url=settings.OPENALEX_BASE_URL,
            api_key=settings.OPENALEX_API_KEY,
            mailto=settings.OPENALEX_MAILTO,
            timeout=settings.HTTP_TIMEOUT_SECONDS,
            max_retries=settings.HTTP_MAX_RETRIES,
            backoff_factor=settings.HTTP_RETRY_BACKOFF_FACTOR,
        ),
    }


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search Scholarly Sources",
    description="Retrieve canonical, normalized, deduplicated, and hybrid-ranked ResearchDocuments from scholarly sources.",
)
async def search_papers(
    request: SearchRequest,
    adapters: Dict[str, BaseSourceAdapter] = Depends(get_search_adapters),
    cache: InMemoryCache = Depends(get_cache),
    pipeline: HybridRetrievalPipeline = Depends(get_retrieval_pipeline),
    metrics: MetricsCollector = Depends(get_metrics_collector),
    session: AsyncSession = Depends(get_db_session),
) -> SearchResponse:
    """Execute multi-source search, normalize records, deduplicate, cache, and rank via hybrid retrieval."""
    metrics.increment_search_requests()

    # Determine which adapters to query
    target_names = request.sources if request.sources is not None else list(adapters.keys())

    # Validate requested sources
    invalid_sources = [name for name in target_names if name not in adapters]
    if invalid_sources:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported source(s): {', '.join(invalid_sources)}. Supported: {list(adapters.keys())}",
        )

    provider_status: Dict[str, str] = {}
    cached_status: Dict[str, bool] = {}
    collected_papers: List[SourcePaper] = []

    # Check cache first for each target provider
    needed_adapters: Dict[str, BaseSourceAdapter] = {}
    for name in target_names:
        cache_key = generate_cache_key(
            provider=name,
            query=request.query,
            pagination={"limit": request.limit},
        )
        cached_data = await cache.get(cache_key)
        if cached_data is not None:
            logger.debug("[%s] Cache hit for query: '%s'", name, request.query)
            metrics.record_cache_access(True)
            cached_status[name] = True
            provider_status[name] = "ok"
            collected_papers.extend(cached_data)
        else:
            metrics.record_cache_access(False)
            cached_status[name] = False
            needed_adapters[name] = adapters[name]

    # Concurrently execute queries for cache misses
    if needed_adapters:
        names = list(needed_adapters.keys())
        source_start = time.perf_counter()
        tasks = [needed_adapters[name].search(request.query, request.limit) for name in names]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        source_latency_ms = (time.perf_counter() - source_start) * 1000.0
        metrics.record_latency("source_latency", source_latency_ms)

        for name, result in zip(names, task_results):
            if isinstance(result, Exception):
                error_msg = f"Failed: {type(result).__name__} - {result}"
                logger.error("[%s] Search failed: %s", name, error_msg)
                provider_status[name] = f"error: {type(result).__name__}"
                metrics.record_provider_error(name, type(result).__name__)
            else:
                provider_status[name] = "ok"
                collected_papers.extend(result)

                # Cache successful results
                cache_key = generate_cache_key(
                    provider=name,
                    query=request.query,
                    pagination={"limit": request.limit},
                )
                await cache.set(cache_key, result)

        # Close clients if owned
        for adapter in needed_adapters.values():
            await adapter.close()

    # Step 1: Normalize intermediate SourcePapers to ResearchDocuments
    normalized_docs: List[ResearchDocument] = []
    for paper in collected_papers:
        doc = normalize_source_paper(paper)
        if doc is not None:
            normalized_docs.append(doc)

    # Step 2: Deduplicate documents preserving provenance
    canonical_documents = deduplicate_documents(normalized_docs)

    # Step 3: Hybrid Retrieval (BM25 + SBERT -> RRF -> Bounded candidate pool -> Cross-Encoder)
    retrieval_start = time.perf_counter()
    candidates = await pipeline.retrieve(
        query=request.query,
        documents=canonical_documents,
        candidate_limit=request.limit,
    )
    retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0
    metrics.record_latency("retrieval_latency", retrieval_latency_ms)
    metrics.record_candidate_count(len(candidates))

    if (
        candidates
        and candidates[0].rerank_metadata
        and candidates[0].rerank_metadata.model_metadata
        and candidates[0].rerank_metadata.model_metadata.latency_ms is not None
    ):
        metrics.record_latency("reranking_latency", candidates[0].rerank_metadata.model_metadata.latency_ms)

    # Step 4: Formal Utility Model (Phase 6 - Optional User Preferences, Feedback & Constraints)
    run_id: Optional[str] = None
    if request.preferences or request.feedback or request.constraints or request.weights:
        from app.feedback.models import FeedbackItem
        from app.feedback.validator import FeedbackContradictionError
        from app.optimization.constraints import ConstraintContradictionError, SearchConstraints
        from app.optimization.utility import FormalUtilityModel, UtilityWeights
        from app.preferences.models import UserPreferences

        prefs = UserPreferences(**request.preferences) if isinstance(request.preferences, dict) else request.preferences
        cons = SearchConstraints(**request.constraints) if isinstance(request.constraints, dict) else request.constraints
        w = UtilityWeights(**request.weights) if isinstance(request.weights, dict) else request.weights
        fb_items = None
        if request.feedback:
            fb_items = [
                FeedbackItem(**item) if isinstance(item, dict) else item
                for item in request.feedback
            ]

        utility_model = FormalUtilityModel()
        try:
            candidates, run_record = utility_model.evaluate(
                candidates=candidates,
                preferences=prefs,
                feedback_items=fb_items,
                constraints=cons,
                weights=w,
                persist=True,
            )
            run_id = run_record.run_id
        except (FeedbackContradictionError, ConstraintContradictionError) as exc:
            logger.warning("Contradiction in search request: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            )

    # Step 5: Global Ranking Optimization & Verification (Phase 7 - Optional)
    optimization_result = None
    if request.optimize:
        from app.optimization.models import ObjectiveType, OptimizationProblem
        from app.optimization.optimizer import GlobalRankingOptimizer

        obj_type = (
            ObjectiveType.PAIRWISE_DIVERSITY
            if request.optimization_objective == "pairwise_diversity"
            else ObjectiveType.LINEAR_POSITIONAL
        )
        opt_prefs = UserPreferences(**request.preferences) if isinstance(request.preferences, dict) else request.preferences
        opt_cons = SearchConstraints(**request.constraints) if isinstance(request.constraints, dict) else request.constraints
        opt_weights = UtilityWeights(**request.weights) if isinstance(request.weights, dict) else request.weights
        opt_fb = [FeedbackItem(**item) if isinstance(item, dict) else item for item in request.feedback] if request.feedback else None

        problem = OptimizationProblem(
            candidates=candidates,
            k=request.limit,
            objective_type=obj_type,
            diversity_lambda=request.diversity_lambda,
            preferences=opt_prefs,
            feedback_items=opt_fb,
            constraints=opt_cons,
            weights=opt_weights,
        )
        opt_start = time.perf_counter()
        optimizer = GlobalRankingOptimizer()
        opt_res = optimizer.optimize(problem)
        opt_latency_ms = (time.perf_counter() - opt_start) * 1000.0
        metrics.record_latency("optimization_latency", opt_latency_ms)
        metrics.record_solver_status(opt_res.solver_used.value, opt_res.solver_status.value)
        candidates = opt_res.ranked_candidates
        optimization_result = opt_res

    ranked_documents = [c.document for c in candidates]

    # Step 6: Research Intelligence Report (Phase 8 - Optional)
    intelligence_report = None
    if request.include_intelligence:
        from app.intelligence.service import ResearchIntelligenceService
        intel_service = ResearchIntelligenceService()
        intelligence_report = intel_service.generate_full_report(
            query=request.query,
            documents=ranked_documents,
            run_id=run_id,
        )

    # Step 7: Persistence to Database (Phase 9)
    try:
        repo = DatabaseRepository(session)
        if canonical_documents:
            await repo.upsert_documents(canonical_documents)
        if optimization_result is not None:
            await repo.save_optimization_result(optimization_result, run_id=run_id)
        if intelligence_report and intelligence_report.roadmap:
            await repo.save_roadmap(intelligence_report.roadmap)
    except Exception as exc:
        logger.warning("Database persistence during search failed gracefully: %s", exc)

    return SearchResponse(
        query=request.query,
        total_results=len(ranked_documents),
        provider_status=provider_status,
        cached=cached_status,
        results=ranked_documents,
        candidates=candidates,
        run_id=run_id,
        optimization_result=optimization_result,
        intelligence=intelligence_report,
    )

