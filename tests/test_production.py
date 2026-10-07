"""
Phase 9 — Production Integration & Comprehensive Verification Tests.

Covers:
1. Database persistence (ResearchDocument, provenance, saved papers, feedback, preferences, runs, roadmap)
2. Vector layer (indexing, cosine search, versioning, serialization, rebuild)
3. Observability (latency tracking, cache hit rates, provider errors, candidate counts, solver status)
4. Evaluation (Precision@K, Recall@K, NDCG@K, verification rate, buildability evaluation)
5. Connected frontend APIs (paper details, save/remove, feedback, preferences, health check)
6. End-to-end pipeline execution verifying the exact conditional optimality claim.
"""

import asyncio
import os
import shutil
import tempfile
from typing import Dict, List
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.search import get_search_adapters
from app.config import get_settings
from app.db.models import Base
from app.db.repository import DatabaseRepository
from app.evaluation.metrics import (
    evaluate_buildability_accuracy,
    ndcg_at_k,
    optimization_verification_rate,
    precision_at_k,
    recall_at_k,
    search_success_rate,
    user_satisfaction_score,
)
from app.feedback.models import FeedbackItem, FeedbackType
from app.ingestion.cache import get_cache
from app.ingestion.deduplicate import deduplicate_documents
from app.ingestion.normalize import normalize_source_paper
from app.intelligence.service import ResearchIntelligenceService
from app.main import app
from app.observability.metrics import MetricsCollector
from app.optimization.constraints import SearchConstraints
from app.optimization.models import ObjectiveType, OptimizationProblem
from app.optimization.optimizer import GlobalRankingOptimizer
from app.optimization.utility import FormalUtilityModel, UtilityWeights
from app.preferences.models import UserPreferences
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.pipeline import HybridRetrievalPipeline
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.rrf import RRFCombiner
from app.retrieval.sbert import SBERTRetriever
from app.schemas.paper import ProvenanceRecord, ResearchDocument
from app.schemas.source import SearchRequest, SourcePaper
from app.sources.base import BaseSourceAdapter
from app.vector.index import VectorIndex

client = TestClient(app)
settings = get_settings()


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Ensure database tables exist and search cache is clear before every test."""
    from app.db.session import init_db
    asyncio.run(init_db())
    asyncio.run(get_cache().clear())


@pytest.fixture
def in_memory_db_session():
    """Create an isolated SQLite in-memory database session for tests."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async_session = async_sessionmaker(engine, expire_on_commit=False)

    async def _setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_setup())

    async def _get_session():
        async with async_session() as session:
            yield session

    return _get_session, engine


def test_database_repository_crud(in_memory_db_session):
    """Test full async CRUD operations on DatabaseRepository."""
    get_session_gen, engine = in_memory_db_session

    async def _run():
        async for session in get_session_gen():
            repo = DatabaseRepository(session)

            # 1. ResearchDocument & Provenance
            doc = ResearchDocument(
                internal_id="doc_test_1",
                source="test_source",
                source_id="101",
                title="Foundations of Scalable Transformers",
                abstract="A study on scaling laws.",
                authors=["Alice Smith", "Bob Jones"],
                publication_year=2023,
                citation_count=42,
                provenance=[
                    ProvenanceRecord(
                        source="test_source",
                        source_id="101",
                        retrieval_timestamp="2026-10-07T00:00:00Z",
                    )
                ],
            )
            await repo.upsert_document(doc)
            fetched = await repo.get_document("doc_test_1")
            assert fetched is not None
            assert fetched.title == "Foundations of Scalable Transformers"
            assert len(fetched.provenance) == 1

            # 2. Saved Papers
            saved = await repo.save_paper("doc_test_1", user_id="user_1", notes="Crucial paper", tags=["nlp"])
            assert saved.document_id == "doc_test_1"
            papers = await repo.get_saved_papers(user_id="user_1")
            assert len(papers) == 1
            assert papers[0].notes == "Crucial paper"

            removed = await repo.remove_saved_paper("doc_test_1", user_id="user_1")
            assert removed is True
            assert len(await repo.get_saved_papers(user_id="user_1")) == 0

            # 3. User Feedback
            fb = FeedbackItem(
                document_id="doc_test_1",
                feedback_type=FeedbackType.RELEVANT,
                rationale="Highly aligned with NLP goals",
            )
            await repo.save_feedback(fb, user_id="user_1")
            fb_list = await repo.get_feedback_items(user_id="user_1")
            assert len(fb_list) == 1
            assert fb_list[0].feedback_type == FeedbackType.RELEVANT

            # 4. User Preferences
            prefs = UserPreferences(
                relevance_weight=0.4,
                recency_weight=0.3,
                source_code_availability_weight=0.3,
            )
            await repo.save_preferences(prefs, user_id="user_1")
            fetched_prefs = await repo.get_preferences(user_id="user_1")
            assert fetched_prefs is not None
            assert fetched_prefs.recency_weight == 0.3

            # 5. Research Roadmap
            intel_service = ResearchIntelligenceService()
            report = intel_service.generate_full_report(
                query="Transformer Scaling",
                documents=[doc],
            )
            assert report.roadmap is not None
            await repo.save_roadmap(report.roadmap)
            fetched_roadmap = await repo.get_roadmap(report.roadmap.roadmap_id)
            assert fetched_roadmap is not None
            assert fetched_roadmap.title == report.roadmap.title
            assert len(fetched_roadmap.stages) > 0

    asyncio.run(_run())


def test_vector_index_operations():
    """Test VectorIndex indexing, cosine similarity, versioning, and rebuilding."""
    temp_dir = tempfile.mkdtemp()
    try:
        index = VectorIndex(dimension=4, model_version="test-model-v1", index_version="1.0.0")

        # 3 vectors
        v1 = np.array([1.0, 0.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0, 0.0])
        v3 = np.array([0.707, 0.707, 0.0, 0.0])

        index.add("doc_1", v1)
        index.add("doc_2", v2)
        index.add("doc_3", v3)
        assert index.size == 3

        # Query aligned with v1
        results = index.search(np.array([1.0, 0.0, 0.0, 0.0]), top_k=2)
        assert len(results) == 2
        assert results[0][0] == "doc_1"
        assert abs(results[0][1] - 1.0) < 1e-4

        # Serialization to disk
        index.save(temp_dir)
        loaded = VectorIndex.load(temp_dir)
        assert loaded.size == 3
        assert loaded.model_version == "test-model-v1"
        assert loaded.index_version == "1.0.0"

        # Rebuild capability
        doc_a = ResearchDocument(
            internal_id="doc_a",
            source="src",
            source_id="a",
            title="Alpha",
            abstract="Alpha doc",
        )
        doc_b = ResearchDocument(
            internal_id="doc_b",
            source="src",
            source_id="b",
            title="Beta",
            abstract="Beta doc",
        )

        def mock_embed(text: str) -> np.ndarray:
            return np.array([1.0, 0.0, 0.0, 0.0]) if "Alpha" in text else np.array([0.0, 1.0, 0.0, 0.0])

        loaded.rebuild([doc_a, doc_b], mock_embed)
        assert loaded.size == 2
        rebuilt_res = loaded.search(np.array([1.0, 0.0, 0.0, 0.0]), top_k=1)
        assert rebuilt_res[0][0] == "doc_a"

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_observability_metrics_collection():
    """Test MetricsCollector telemetry gathering and statistics computation."""
    collector = MetricsCollector()
    collector.clear()

    collector.increment_search_requests()
    collector.record_latency("source_latency", 45.2)
    collector.record_latency("source_latency", 54.8)
    collector.record_latency("retrieval_latency", 12.0)
    collector.record_latency("reranking_latency", 25.5)
    collector.record_latency("optimization_latency", 8.4)

    collector.record_cache_access(True)
    collector.record_cache_access(False)
    collector.record_cache_access(True)

    collector.record_provider_error("openalex", "RateLimitError")
    collector.record_candidate_count(20)
    collector.record_candidate_count(15)
    collector.record_solver_status("HungarianAssignmentSolver", "OPTIMAL")

    summary = collector.get_summary()

    assert summary["total_search_requests"] == 1
    assert summary["cache"]["hits"] == 2
    assert summary["cache"]["misses"] == 1
    assert summary["cache"]["hit_rate"] == 0.6667
    assert summary["provider_errors"]["openalex"]["RateLimitError"] == 1
    assert summary["candidate_pool_stats"]["mean_candidates"] == 17.5
    assert summary["latencies"]["source_latency"]["count"] == 2
    assert summary["latencies"]["source_latency"]["mean_ms"] == 50.0
    assert summary["solver_statuses"]["HungarianAssignmentSolver:OPTIMAL"] == 1


def test_evaluation_metrics_and_benchmarks():
    """Test Precision@K, Recall@K, NDCG@K, verification rate, and buildability accuracy."""
    retrieved = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
    relevant = {"doc_1", "doc_3", "doc_6"}

    # Precision@3: doc_1 and doc_3 are relevant -> 2/3
    p3 = precision_at_k(retrieved, relevant, k=3)
    assert abs(p3 - (2.0 / 3.0)) < 1e-4

    # Recall@3: 2 found out of 3 total relevant -> 2/3
    r3 = recall_at_k(retrieved, relevant, k=3)
    assert abs(r3 - (2.0 / 3.0)) < 1e-4

    # Recall with empty ground truth
    assert recall_at_k(retrieved, set(), k=3) == 0.0

    # NDCG@3 with binary relevance
    ndcg3 = ndcg_at_k(retrieved, relevant, k=3)
    assert 0.0 < ndcg3 <= 1.0

    # Search success rate
    ssr = search_success_rate([["d1", "d2"], ["d3"]], [{"d1"}, {"d4"}], k=2)
    assert ssr == 0.5

    # User satisfaction score
    fb_pos = FeedbackItem(document_id="d1", feedback_type=FeedbackType.RELEVANT)
    fb_neg = FeedbackItem(document_id="d2", feedback_type=FeedbackType.NOT_RELEVANT)
    satisfaction = user_satisfaction_score([fb_pos, fb_pos, fb_neg])
    assert abs(satisfaction - (2.0 / 3.0)) < 1e-4

    # Optimization verification rate
    class MockVerification:
        def __init__(self, is_v: bool, cons_sat: bool):
            self.is_verified = is_v
            self.constraints_satisfied = cons_sat

    ovr = optimization_verification_rate([
        MockVerification(True, True),
        MockVerification(False, True),
        MockVerification(True, True),
    ])  # type: ignore
    assert abs(ovr - (2.0 / 3.0)) < 1e-4

    # Buildability evaluation against ground truth
    class MockBuildability:
        def __init__(self, doc_id: str, score: float, tier: str = "Medium"):
            self.document_id = doc_id
            self.overall_score = score
            self.reproducibility_tier = tier

    reports = [MockBuildability("d1", 0.8, "High"), MockBuildability("d2", 0.4, "Low")]  # type: ignore
    ground_truth = {
        "d1": {"overall_score": 0.85, "reproducibility_tier": "High"},
        "d2": {"overall_score": 0.35, "reproducibility_tier": "Low"},
    }
    b_eval = evaluate_buildability_accuracy(reports, ground_truth)
    assert b_eval["evaluated_count"] == 2
    assert b_eval["mean_absolute_error"] < 0.1
    assert b_eval["tier_accuracy"] == 1.0


def test_connected_frontend_endpoints():
    """Verify frontend API endpoints under /api/v1."""
    # 1. Health check with database
    health_resp = client.get("/api/v1/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    assert health_data["status"] == "ok"
    assert "database" in health_data

    # 2. Database direct health
    db_health = client.get("/api/v1/health/db")
    assert db_health.status_code == 200
    assert db_health.json()["status"] == "connected"

    # 3. Observability metrics endpoint
    metrics_resp = client.get("/api/v1/observability/metrics")
    assert metrics_resp.status_code == 200
    assert "latencies" in metrics_resp.json()

    # 4. Evaluation benchmark endpoint
    bench_payload = {
        "retrieved_ids": ["doc_1", "doc_2", "doc_3"],
        "ground_truth_relevant_ids": ["doc_1", "doc_3"],
        "k": 3,
    }
    bench_resp = client.post("/api/v1/evaluation/benchmark", json=bench_payload)
    assert bench_resp.status_code == 200
    b_data = bench_resp.json()
    assert b_data["k"] == 3
    assert b_data["precision_at_k"] > 0.0

    # 5. Saved papers endpoint
    # Pre-seed document so save_paper finds it
    from app.db.session import get_session_factory
    factory = get_session_factory()

    async def _seed_test_doc():
        async with factory() as s:
            r = DatabaseRepository(s)
            await r.upsert_document(
                ResearchDocument(
                    internal_id="doc_frontend_test",
                    source="test_src",
                    source_id="frontend_1",
                    title="Frontend Interface Integration Paper",
                    abstract="Test paper for saved library.",
                )
            )

    asyncio.run(_seed_test_doc())

    # Paper details endpoint check
    doc_detail_resp = client.get("/api/v1/papers/doc_frontend_test")
    assert doc_detail_resp.status_code == 200
    assert doc_detail_resp.json()["internal_id"] == "doc_frontend_test"

    save_payload = {
        "document_id": "doc_frontend_test",
        "user_id": "test_user_ui",
        "notes": "Testing saved paper flow",
        "tags": ["frontend", "nlp"],
    }
    save_resp = client.post("/api/v1/papers/saved", json=save_payload)
    assert save_resp.status_code == 201
    assert save_resp.json()["document_id"] == "doc_frontend_test"

    list_saved = client.get("/api/v1/papers/saved?user_id=test_user_ui")
    assert list_saved.status_code == 200
    assert len(list_saved.json()) >= 1

    del_resp = client.delete("/api/v1/papers/saved/doc_frontend_test?user_id=test_user_ui")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    # 6. Feedback endpoint
    fb_payload = {
        "user_id": "test_user_ui",
        "document_id": "doc_frontend_test",
        "feedback_type": "relevant",
        "rationale": "High quality paper",
    }
    fb_resp = client.post("/api/v1/feedback", json=fb_payload)
    assert fb_resp.status_code == 201

    get_fb = client.get("/api/v1/feedback?user_id=test_user_ui")
    assert get_fb.status_code == 200
    assert len(get_fb.json()) >= 1

    # 7. Preferences endpoint
    pref_payload = {
        "user_id": "test_user_ui",
        "preferences": {
            "relevance_weight": 0.5,
            "recency_weight": 0.2,
            "source_code_availability_weight": 0.3,
        },
    }
    put_pref = client.put("/api/v1/preferences", json=pref_payload)
    assert put_pref.status_code == 200

    get_pref = client.get("/api/v1/preferences?user_id=test_user_ui")
    assert get_pref.status_code == 200
    assert get_pref.json()["relevance_weight"] == 0.5


def test_full_pipeline_end_to_end():
    """
    Complete end-to-end integration test validating the entire ResearchMind AI architecture:
    Query → sources → normalize → deduplicate → BM25/SBERT → RRF → Cross-Encoder
    → feedback/preferences/constraints → utility → optimization → explanation
    → buildability → graph → roadmap.

    Strictly preserves the conditional optimality guarantee.
    """
    # 1. Mock Scholarly Sources
    papers_s2 = [
        SourcePaper(
            source="semantic_scholar",
            source_id="s2_scale_1",
            title="Scaling Laws for Neural Language Models",
            abstract="We investigate empirical scaling laws for language model performance with compute and datasets.",
            authors=["Jared Kaplan", "Sam McCandlish"],
            year=2020,
            citation_count=3500,
            doi="10.48550/arXiv.2001.08361",
            raw_metadata={"isOpenAccess": True, "code_url": "https://github.com/scaling/nlp"},
        ),
        SourcePaper(
            source="semantic_scholar",
            source_id="s2_scale_2",
            title="Chinchilla: Training Compute-Optimal Large Language Models",
            abstract="We find that for compute-optimal training, dataset size and model size should be scaled equally.",
            authors=["Jordan Hoffmann", "Sebastian Borgeaud"],
            year=2022,
            citation_count=2100,
            doi="10.48550/arXiv.2203.15556",
            raw_metadata={"isOpenAccess": True, "code_url": "https://github.com/deepmind/chinchilla"},
        ),
    ]

    papers_oa = [
        SourcePaper(
            source="openalex",
            source_id="oa_scale_1",
            title="Scaling Laws for Neural Language Models",
            abstract="Empirical scaling study from OpenAI.",
            authors=["Jared Kaplan"],
            year=2020,
            citation_count=3490,
            doi="10.48550/arXiv.2001.08361",  # Same DOI -> will deduplicate
            raw_metadata={"open_access": {"is_oa": True}},
        ),
        SourcePaper(
            source="openalex",
            source_id="oa_scale_3",
            title="Llama: Open and Efficient Foundation Language Models",
            abstract="A collection of foundation models ranging from 7B to 65B parameters trained on publicly available datasets.",
            authors=["Hugo Touvron", "Thibaut Lavril"],
            year=2023,
            citation_count=4500,
            raw_metadata={"open_access": {"is_oa": True}, "code_url": "https://github.com/facebookresearch/llama"},
        ),
    ]

    class MockAdapter(BaseSourceAdapter):
        def __init__(self, p_list):
            super().__init__(base_url="http://mock")
            self.p_list = p_list

        async def search(self, query: str, limit: int = 10):
            return self.p_list[:limit]

    adapters = {
        "semantic_scholar": MockAdapter(papers_s2),
        "openalex": MockAdapter(papers_oa),
    }

    app.dependency_overrides[get_search_adapters] = lambda: adapters

    try:
        # Full query with preferences, feedback, constraints, optimization, and intelligence
        request_body = {
            "query": "scaling laws for language models",
            "limit": 5,
            "preferences": {
                "relevance_weight": 0.35,
                "recency_weight": 0.20,
                "source_code_availability_weight": 0.25,
                "pretrained_model_availability_weight": 0.20,
            },
            "feedback": [
                {
                    "document_id": "s2_scale_2",
                    "feedback_type": "prefer_similar",
                    "notes": "High practical value compute-optimal guidelines",
                }
            ],
            "constraints": {
                "require_open_access": True,
            },
            "weights": {
                "relevance": 0.35,
                "preference_match": 0.25,
                "feedback_match": 0.20,
                "buildability": 0.10,
                "source_quality": 0.05,
                "recency": 0.05,
            },
            "optimize": True,
            "optimization_objective": "linear_positional",
            "include_intelligence": True,
        }

        resp = client.post("/api/v1/search", json=request_body)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        # Step A: Validate Normalization & Deduplication
        # Out of 4 papers, 2 share the same DOI (Kaplan et al.), so total unique docs should be 3
        assert data["total_results"] == 3
        results = data["results"]
        assert len(results) == 3

        # Step B: Validate Candidates with RRF and Utility
        candidates = data["candidates"]
        assert len(candidates) == 3
        for c in candidates:
            assert c["metadata"]["rrf_score"] > 0
            assert "document" in c

        # Step C: Validate Global Optimization & Conditional Optimality Guarantee
        opt_res = data["optimization_result"]
        assert opt_res is not None
        assert opt_res["verification"]["is_verified"] is True
        assert opt_res["verification"]["constraints_satisfied"] is True
        # Exact non-weakened conditional optimality claim
        expected_claim = "globally optimal over the retrieved candidate set under the explicitly defined utility and constraints"
        assert opt_res["guarantee_scope"] == expected_claim

        # Step D: Validate Research Intelligence (Explanation, Buildability, Graph, Roadmap)
        intel = data["intelligence"]
        assert intel is not None

        # 1. Explainability
        assert "explanations" in intel
        assert len(intel["explanations"]) == 3
        first_exp = intel["explanations"][0]
        assert "components" in first_exp
        assert "narrative_explanation" in first_exp

        # 2. Buildability
        assert "buildability_reports" in intel
        assert len(intel["buildability_reports"]) == 3
        b_report = intel["buildability_reports"][0]
        assert 0.0 <= b_report["overall_score"] <= 1.0
        assert "dimensions" in b_report
        assert "dataset_availability" in b_report["dimensions"]
        assert "source_code_availability" in b_report["dimensions"]

        # 3. Knowledge Graph
        kg = intel["knowledge_graph"]
        assert kg is not None
        assert len(kg["nodes"]) >= 3
        node_types = {n["type"] for n in kg["nodes"]}
        assert "paper" in node_types

        # 4. Roadmap
        roadmap = intel["roadmap"]
        assert roadmap is not None
        assert roadmap["title"] != ""
        assert len(roadmap["stages"]) >= 4
        stage_types = [s["stage"] for s in roadmap["stages"]]
        assert "dataset_preparation" in stage_types
        assert "model_training" in stage_types
        assert "deployment" in stage_types

        # Step E: Verify Persistent Storage
        # Document details should be queryable via GET /api/v1/papers/{id}
        doc_id = results[0]["internal_id"]
        paper_detail_resp = client.get(f"/api/v1/papers/{doc_id}")
        assert paper_detail_resp.status_code == 200
        assert paper_detail_resp.json()["internal_id"] == doc_id

    finally:
        app.dependency_overrides.clear()
