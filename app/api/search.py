import asyncio
import logging
import math
import time
import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db.repository import DatabaseRepository
from app.db.session import get_db_session
from app.ingestion.cache import InMemoryCache, generate_cache_key, get_cache
from app.ingestion.deduplicate import deduplicate_documents
from app.ingestion.normalize import normalize_source_paper
from app.observability.context import get_current_request_id
from app.observability.logging import log_pipeline_telemetry
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
    pipeline_total_start = time.perf_counter()
    active_request_id = get_current_request_id() or uuid.uuid4().hex
    metrics.increment_search_requests()

    pipeline_errors: List[str] = []

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

    # Concurrently execute queries for cache misses with strict failure isolation
    source_latency_ms = 0.0
    if needed_adapters:
        names = list(needed_adapters.keys())
        source_start = time.perf_counter()
        tasks = [needed_adapters[name].search(request.query, request.limit) for name in names]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        source_latency_ms = (time.perf_counter() - source_start) * 1000.0
        metrics.record_latency("source_retrieval_latency", source_latency_ms)
        metrics.record_latency("source_latency", source_latency_ms)

        for name, result in zip(names, task_results):
            if isinstance(result, Exception):
                error_msg = f"Failed: {type(result).__name__} - {result}"
                logger.error("[%s] Search failed for req_id=%s: %s", name, active_request_id, error_msg)
                provider_status[name] = f"error: {type(result).__name__}"
                metrics.record_source_failure(name, type(result).__name__)
                pipeline_errors.append(f"{name}: {type(result).__name__}")
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

    # Step 1: Normalize intermediate SourcePapers to ResearchDocuments & Step 2: Deduplicate
    norm_start = time.perf_counter()
    normalized_docs: List[ResearchDocument] = []
    for paper in collected_papers:
        doc = normalize_source_paper(paper)
        if doc is not None:
            normalized_docs.append(doc)

    canonical_documents = deduplicate_documents(normalized_docs)
    norm_latency_ms = (time.perf_counter() - norm_start) * 1000.0
    metrics.record_latency("normalization_latency", norm_latency_ms)

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

    stage_latencies = getattr(pipeline, "last_latencies", {})

    target_limit = request.top_k if request.top_k is not None else request.limit

    # Step 4: Paper-Specific Buildability Evaluation across 9 Dimensions
    bld_start = time.perf_counter()
    from app.intelligence.buildability import BuildabilityEvaluator
    bld_eval = BuildabilityEvaluator()
    for cand in candidates:
        rep = bld_eval.evaluate_document(cand.document)
        cand.document.buildability_score = rep.overall_score
        cand.document.buildability = rep.overall_score
        cand.document.buildability_breakdown = rep.evidence_breakdown
        cand.document.evidence_sources = rep.evidence_sources
        cand.document.unknown_fields = rep.unknown_fields
        if cand.features is None:
            cand.features = {}
        cand.features["buildability"] = rep.overall_score
    bld_latency_ms = (time.perf_counter() - bld_start) * 1000.0
    metrics.record_latency("buildability_latency", bld_latency_ms)

    # Step 5: Formal Multi-Criteria Utility Model (Single Source of Truth)
    from app.feedback.models import FeedbackItem
    from app.feedback.validator import FeedbackContradictionError
    from app.optimization.constraints import ConstraintContradictionError, SearchConstraints
    from app.optimization.utility import FormalUtilityModel, UtilityWeights
    from app.preferences.models import UserPreferences

    prefs = (
        UserPreferences(**request.preferences)
        if isinstance(request.preferences, dict)
        else (request.preferences or UserPreferences())
    )
    cons = (
        SearchConstraints(**request.constraints)
        if isinstance(request.constraints, dict)
        else (request.constraints or SearchConstraints())
    )
    w = (
        UtilityWeights(**request.weights)
        if isinstance(request.weights, dict)
        else (request.weights or UtilityWeights())
    )
    fb_items = None
    if request.feedback:
        fb_items = [
            FeedbackItem(**item) if isinstance(item, dict) else item
            for item in request.feedback
        ]

    utility_model = FormalUtilityModel(default_weights=w)
    run_id: Optional[str] = None
    run_record = None

    util_start = time.perf_counter()
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
        logger.warning("Contradiction in search request [req_id=%s]: %s", active_request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    finally:
        util_latency_ms = (time.perf_counter() - util_start) * 1000.0
        metrics.record_latency("utility_calculation_latency", util_latency_ms)

    # Step 6: Global Ranking Optimization & Verification
    optimization_result = None
    structured_optimization: Optional[Dict[str, Any]] = None
    opt_latency_ms = 0.0
    verif_latency_ms = 0.0
    should_optimize = request.optimize or bool(request.optimization)

    if should_optimize and candidates:
        from app.optimization.models import ObjectiveType, OptimizationProblem, SolverStatus
        from app.optimization.optimizer import GlobalRankingOptimizer
        from app.optimization.verification import OptimizationVerifier

        opt_dict = request.optimization if isinstance(request.optimization, dict) else {}
        obj_str = opt_dict.get("objective", request.optimization_objective)
        div_lambda = opt_dict.get("diversity_lambda", request.diversity_lambda)

        obj_type = (
            ObjectiveType.PAIRWISE_DIVERSITY
            if obj_str == "pairwise_diversity"
            else ObjectiveType.LINEAR_POSITIONAL
        )

        problem = OptimizationProblem(
            candidates=candidates,
            k=target_limit,
            objective_type=obj_type,
            diversity_lambda=div_lambda,
            preferences=prefs,
            feedback_items=fb_items,
            constraints=cons,
            weights=w,
        )

        opt_start = time.perf_counter()
        try:
            verifier = OptimizationVerifier(exhaustive_threshold=10 if request.verification else 0)
            optimizer = GlobalRankingOptimizer(utility_model=utility_model, verifier=verifier)
            opt_res = optimizer.optimize(problem)
            opt_latency_ms = (time.perf_counter() - opt_start) * 1000.0
            metrics.record_latency("optimization_latency", opt_latency_ms)
            metrics.record_solver_status(opt_res.solver_used.value, opt_res.solver_status.value)
            candidates = opt_res.ranked_candidates
            optimization_result = opt_res

            verif = opt_res.verification
            if verif and hasattr(verif, "verification_time_ms") and verif.verification_time_ms is not None:
                verif_latency_ms = verif.verification_time_ms
                metrics.record_latency("verification_latency", verif_latency_ms)

            if verif and not verif.is_verified and request.verification:
                metrics.record_verification_failure()

            status_label = opt_res.optimization_status.value if hasattr(opt_res, "optimization_status") else "globally_optimized"

            structured_optimization = {
                "method": opt_res.solver_used.value,
                "status": status_label,
                "optimality_gap": verif.approximation_gap if (verif and verif.approximation_gap is not None) else 0.0,
                "verified": verif.is_verified if verif else False,
                "guarantee_scope": opt_res.guarantee_scope,
                "reported_objective": opt_res.objective_value,
                "exhaustive_comparison": verif.exhaustive_comparison_performed if verif else False,
            }
        except Exception as opt_err:
            metrics.record_optimization_failure()
            pipeline_errors.append(f"optimization: {type(opt_err).__name__}")
            logger.error("Optimization failed for run_id=%s, req_id=%s: %s", run_id, active_request_id, opt_err)
            structured_optimization = {
                "method": "fallback_utility",
                "status": "failed",
                "error": str(opt_err),
            }

    # Step 7: Grounded Transparent Explainability via RankingExplainer
    from app.intelligence.explainer import RankingExplainer
    explainer = RankingExplainer()
    for rank_idx, cand in enumerate(candidates, start=1):
        doc = cand.document
        doc.rank = rank_idx
        doc.paper_id = doc.internal_id
        doc.buildability_evidence = doc.buildability_breakdown
        if run_record:
            try:
                explanation_obj = explainer.explain_candidate(
                    run_record=run_record,
                    document_id=doc.internal_id,
                    rank=rank_idx,
                )
                doc.explanation = explanation_obj.narrative_explanation
                if "relevance" in explanation_obj.components:
                    doc.relevance_score = explanation_obj.components["relevance"].raw_score
                if "preference_match" in explanation_obj.components:
                    doc.preference_contribution = explanation_obj.components["preference_match"].weighted_score
                if "feedback_match" in explanation_obj.components:
                    doc.feedback_contribution = explanation_obj.components["feedback_match"].weighted_score
                if "penalties" in explanation_obj.components:
                    doc.constraint_penalty = abs(explanation_obj.components["penalties"].weighted_score)
                doc.utility_score = explanation_obj.final_utility_score
            except Exception as ex:
                logger.debug("Explanation attribution for %s: %s", doc.internal_id, ex)
                if not doc.explanation:
                    doc.explanation = f"Ranked #{rank_idx} under multi-criteria utility U(R)."

    ranked_documents = [c.document for c in candidates]

    # Diagnostic output for top papers
    def _extract_score(val):
        if isinstance(val, dict):
            return val.get("score", 0.0)
        return val if val is not None else 0.0

    for idx, cand in enumerate(candidates[:5], start=1):
        doc = cand.document
        bld_pct = (doc.buildability_score or 0.0) * 100.0
        bd = doc.buildability_breakdown or {}
        diag_msg = (
            f"Paper {idx}: {bld_pct:.1f}\n"
            f"  title: {doc.title}\n"
            f"  source: {doc.source}\n"
            f"  utility: {doc.utility_score}\n"
            f"  relevance: {doc.relevance_score}\n"
            f"  dataset: {_extract_score(bd.get('dataset'))}\n"
            f"  code: {_extract_score(bd.get('code'))}\n"
            f"  model: {_extract_score(bd.get('pretrained_model'))}\n"
            f"  sources_count: {len(doc.evidence_sources)}\n"
            f"  unknown_count: {len(doc.unknown_fields)}"
        )
        logger.info(diag_msg)
        try:
            print(diag_msg.encode('ascii', errors='replace').decode('ascii'), flush=True)
        except Exception:
            pass

    # Step 8: Knowledge Graph & Intelligence Synthesis
    intelligence_report = None
    if request.include_intelligence:
        from app.intelligence.service import ResearchIntelligenceService
        intel_service = ResearchIntelligenceService()
        intelligence_report = intel_service.generate_full_report(
            query=request.query,
            documents=ranked_documents,
            run_id=run_id,
        )

    from app.intelligence.graph import KnowledgeGraphBuilder
    knowledge_graph = KnowledgeGraphBuilder().build_graph(ranked_documents)

    # Step 9: Persistence to Database
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

    total_pipeline_ms = (time.perf_counter() - pipeline_total_start) * 1000.0

    # Structured pipeline telemetry logging
    log_pipeline_telemetry(
        query=request.query,
        request_id=active_request_id,
        run_id=run_id,
        sources=target_names,
        provider_status=provider_status,
        cached_status=cached_status,
        papers_collected=len(collected_papers),
        result_count=len(ranked_documents),
        candidate_count=len(candidates),
        latencies_ms={
            "source_retrieval_latency": source_latency_ms,
            "normalization_latency": norm_latency_ms,
            "retrieval_latency": retrieval_latency_ms,
            "bm25_latency": stage_latencies.get("bm25_latency", 0.0),
            "sbert_latency": stage_latencies.get("sbert_latency", 0.0),
            "rrf_latency": stage_latencies.get("rrf_latency", 0.0),
            "cross_encoder_latency": stage_latencies.get("cross_encoder_latency", 0.0),
            "buildability_latency": bld_latency_ms,
            "utility_calculation_latency": util_latency_ms,
            "optimization_latency": opt_latency_ms,
            "verification_latency": verif_latency_ms,
            "total_request_latency": total_pipeline_ms,
        },
        optimization_info=structured_optimization,
        errors=pipeline_errors,
    )

    return SearchResponse(
        query=request.query,
        total_results=len(ranked_documents),
        provider_status=provider_status,
        cached=cached_status,
        results=ranked_documents,
        papers=ranked_documents,
        candidates=candidates,
        run_id=run_id,
        request_id=active_request_id,
        optimization=structured_optimization,
        optimization_result=optimization_result,
        intelligence=intelligence_report,
        knowledge_graph=knowledge_graph,
    )


