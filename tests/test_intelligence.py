"""
Comprehensive tests for Phase 8 — Research Intelligence.

Tests cover:
1. Explanation consistency (grounded strictly in stored ranking inputs)
2. Buildability evaluation across all 9 dimensions with evidence
3. Knowledge graph multi-entity representation with relationship provenance
4. 7-stage engineering roadmap linked to supporting research/resources
5. Full API endpoint verification under /api/v1/intelligence
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.feedback.models import FeedbackItem, FeedbackType
from app.intelligence.buildability import BuildabilityEvaluator
from app.intelligence.explainer import RankingExplainer
from app.intelligence.graph import KnowledgeGraphBuilder
from app.intelligence.models import (
    EdgeType,
    NodeType,
    RoadmapStageType,
)
from app.intelligence.roadmap import RoadmapGenerator
from app.intelligence.service import ResearchIntelligenceService
from app.optimization.constraints import SearchConstraints
from app.optimization.utility import FormalUtilityModel, UtilityWeights
from app.preferences.models import UserPreferences
from app.schemas.paper import ResearchDocument
from app.schemas.retrieval import RetrievalMetadata, RetrievedCandidate

client = TestClient(app)


def make_test_doc(
    doc_id: str,
    title: str,
    year: int = 2023,
    citations: int = 500,
    venue: str = "NeurIPS",
    datasets: list = None,
    models: list = None,
    methods: list = None,
    code_url: str = "https://github.com/lab/awesome-model",
    open_access: bool = True,
    license_name: str = "MIT",
    abstract: str = "A comprehensive deep learning framework utilizing attention and diffusion benchmarks.",
    references: list = None,
    concepts: list = None,
) -> ResearchDocument:
    return ResearchDocument(
        internal_id=doc_id,
        source="semantic_scholar",
        source_id=doc_id,
        title=title,
        abstract=abstract,
        authors=["Alice Smith", "Bob Jones"],
        publication_year=year,
        citation_count=citations,
        venue=venue,
        datasets=["ImageNet-1K", "GLUE"] if datasets is None else datasets,
        models=["Transformer-XL", "ResNet-50"] if models is None else models,
        methods=["Self-Attention", "Gradient Clipping"] if methods is None else methods,
        code_url=code_url,
        open_access=open_access,
        license=license_name,
        references=[] if references is None else references,
        concepts=["deep learning", "natural language processing"] if concepts is None else concepts,
    )


def make_test_candidate(doc: ResearchDocument, rel_score: float = 0.85) -> RetrievedCandidate:
    meta = RetrievalMetadata(
        bm25_rank=1,
        bm25_score=8.5,
        sbert_rank=1,
        sbert_score=rel_score,
        rrf_rank=1,
        rrf_score=0.03,
    )
    return RetrievedCandidate(
        document=doc,
        retrieval_score=rel_score,
        retrieval_rank=1,
        metadata=meta,
    )


# -------------------------------------------------------------------------
# 1. EXPLAINABILITY TESTS
# -------------------------------------------------------------------------

def test_explanation_consistency_with_stored_run():
    """
    Ensure explanation derives strictly and deterministically from stored ranking inputs.
    """
    doc = make_test_doc("doc_exp_1", "Attention Mechanisms in Practice")
    cand = make_test_candidate(doc, rel_score=0.90)

    weights = UtilityWeights(
        w1_relevance=0.40,
        w2_preference_match=0.20,
        w3_feedback_match=0.15,
        w4_buildability=0.15,
        w5_source_quality=0.05,
        w6_recency=0.05,
    )
    prefs = UserPreferences(importance_relevance=0.9, importance_recency=0.6)
    fb = [FeedbackItem(document_id="doc_exp_1", feedback_type=FeedbackType.RELEVANT)]
    cons = SearchConstraints(min_year=2020)

    utility_model = FormalUtilityModel()
    _, run_record = utility_model.evaluate(
        candidates=[cand],
        preferences=prefs,
        feedback_items=fb,
        constraints=cons,
        weights=weights,
        persist=True,
    )

    explainer = RankingExplainer()
    explanation = explainer.explain_candidate(run_record, document_id="doc_exp_1", rank=1)

    assert explanation.document_id == "doc_exp_1"
    assert explanation.run_id == run_record.run_id
    assert explanation.rank == 1

    # Check that all 7 components are present
    expected_components = ["relevance", "preference_match", "feedback_match", "buildability", "source_quality", "recency", "penalties"]
    for c in expected_components:
        assert c in explanation.components
        comp = explanation.components[c]
        assert comp.raw_score >= 0.0
        assert comp.description is not None

    # Verify mathematical consistency
    eval_item = run_record.evaluations[0]
    assert np.isclose(explanation.final_utility_score, eval_item.scores.utility_score, atol=1e-5)
    assert np.isclose(explanation.components["relevance"].weight, weights.w1_relevance)
    assert np.isclose(explanation.components["relevance"].raw_score, eval_item.scores.relevance)
    assert np.isclose(explanation.components["relevance"].weighted_score, weights.w1_relevance * eval_item.scores.relevance)

    # Narrative checks
    assert "Attention Mechanisms in Practice" in explanation.narrative_explanation
    assert "User feedback actively shaped the score" in explanation.narrative_explanation
    assert len(explanation.active_feedback_summary) > 0


def test_explainer_raises_on_unknown_doc_or_run():
    explainer = RankingExplainer()
    with pytest.raises(KeyError):
        explainer.explain_run("non_existent_run_12345")


# -------------------------------------------------------------------------
# 2. BUILDABILITY TESTS
# -------------------------------------------------------------------------

def test_buildability_all_9_dimensions_and_evidence():
    """
    Verify all 9 dimensions are evaluated with concrete factual evidence.
    """
    doc_high = make_test_doc(
        "doc_bld_1",
        "Open SOTA Framework",
        code_url="https://github.com/org/sota-model",
        datasets=["GLUE", "SuperGLUE"],
        models=["BERT-Large", "RoBERTa"],
        methods=["FlashAttention"],
        license_name="Apache-2.0",
        open_access=True,
        abstract="We open-source an efficient model runnable on a single consumer GPU with lightweight checkpoints.",
    )

    evaluator = BuildabilityEvaluator()
    report = evaluator.evaluate_document(doc_high)

    assert report.document_id == "doc_bld_1"
    assert report.overall_score >= 0.75
    assert report.reproducibility_tier == "High"

    required_dims = [
        "dataset_availability",
        "source_code_availability",
        "pretrained_model_availability",
        "documentation",
        "hardware_requirements",
        "dependency_complexity",
        "license_accessibility",
        "reproducibility",
        "implementation_completeness",
    ]
    for d_name in required_dims:
        assert d_name in report.dimensions, f"Missing dimension: {d_name}"
        dim = report.dimensions[d_name]
        assert 0.0 <= dim.score <= 1.0
        assert len(dim.evidence) > 0, f"Evidence must not be empty for {d_name}"

    assert len(report.key_strengths) > 0
    assert any("https://github.com/org/sota-model" in ev for ev in report.dimensions["source_code_availability"].evidence)


def test_buildability_low_tier_detection():
    doc_low = make_test_doc(
        "doc_bld_2",
        "Theoretical Exploration on Huge Clusters",
        code_url=None,
        datasets=[],
        models=[],
        methods=[],
        license_name=None,
        open_access=False,
        abstract="We train an enormous 500B model on a cluster of 1024 TPUs with proprietary confidential customer data.",
    )

    evaluator = BuildabilityEvaluator()
    report = evaluator.evaluate_document(doc_low)

    assert report.overall_score < 0.50
    assert report.reproducibility_tier in ("Low", "Blocked")
    assert len(report.key_risks_or_blockers) > 0
    # Must flag compute or missing code risks
    assert any("reimplementation" in r or "private" in r or "cluster" in r for r in report.key_risks_or_blockers)


# -------------------------------------------------------------------------
# 3. KNOWLEDGE GRAPH TESTS
# -------------------------------------------------------------------------

def test_knowledge_graph_entities_and_provenance():
    """
    Verify Paper -> Dataset, Model, Method, Repo, Related Paper relationships
    with complete provenance.
    """
    doc_1 = make_test_doc(
        "p1", "Foundation Transformers",
        datasets=["WikiText-103", "CommonCrawl"],
        models=["GPT-Mini"],
        methods=["Rotary Embeddings"],
        code_url="https://github.com/org/gpt-mini",
        references=["p2"],
        concepts=["transformers", "attention"],
    )
    doc_2 = make_test_doc(
        "p2", "Attention Optimization",
        datasets=["WikiText-103"],
        models=["GPT-Mini"],
        methods=["FlashAttention"],
        code_url="https://github.com/org/attn-opt",
        concepts=["transformers", "attention"],
    )

    builder = KnowledgeGraphBuilder()
    kg = builder.build_graph([doc_1, doc_2])

    assert kg.total_nodes >= 5
    assert kg.total_edges >= 5

    # Check node types present
    node_types = {n.type for n in kg.nodes}
    assert NodeType.PAPER in node_types
    assert NodeType.DATASET in node_types
    assert NodeType.MODEL in node_types
    assert NodeType.METHOD in node_types
    assert NodeType.REPOSITORY in node_types

    # Check edge types present
    edge_types = {e.type for e in kg.edges}
    assert EdgeType.USES_DATASET in edge_types
    assert EdgeType.USES_MODEL in edge_types
    assert EdgeType.USES_METHOD in edge_types
    assert EdgeType.HAS_REPOSITORY in edge_types
    assert EdgeType.CITES_PAPER in edge_types  # p1 cites p2

    # Check provenance preservation on EVERY node and edge
    for node in kg.nodes:
        assert node.provenance is not None
        assert node.provenance.source_document_id != ""
        assert node.provenance.source_provider != ""
        assert node.provenance.extracted_field != ""

    for edge in kg.edges:
        assert edge.provenance is not None
        assert edge.provenance.source_document_id != ""
        assert edge.provenance.source_provider != ""


# -------------------------------------------------------------------------
# 4. ROADMAP GENERATION TESTS
# -------------------------------------------------------------------------

def test_roadmap_7_stages_and_research_references():
    """
    Verify roadmap produces all 7 ordered stages linked to research resources.
    """
    doc = make_test_doc(
        "doc_rdm_1",
        "Efficient Retrieval with Reranking",
        datasets=["MS-MARCO"],
        models=["MiniLM-L6"],
        code_url="https://github.com/search/fast-reranker",
    )

    generator = RoadmapGenerator()
    roadmap = generator.generate_roadmap(
        query="neural information retrieval",
        documents=[doc],
    )

    assert len(roadmap.stages) == 7
    stage_types = [s.stage for s in roadmap.stages]
    expected_order = [
        RoadmapStageType.DATASET_PREPARATION,
        RoadmapStageType.BASELINE,
        RoadmapStageType.MODEL_TRAINING,
        RoadmapStageType.EVALUATION,
        RoadmapStageType.BACKEND,
        RoadmapStageType.FRONTEND_INTEGRATION,
        RoadmapStageType.DEPLOYMENT,
    ]
    assert stage_types == expected_order

    # Check prerequisites
    assert roadmap.stages[0].prerequisites == []
    assert "ms_dataset_preparation" in roadmap.stages[1].prerequisites
    assert "ms_baseline" in roadmap.stages[2].prerequisites

    # Check supporting research resources linked
    has_linked_resource = False
    for stage in roadmap.stages:
        assert len(stage.deliverables) > 0
        if stage.supporting_resources:
            has_linked_resource = True
            for res in stage.supporting_resources:
                assert res.resource_type in ("paper", "dataset", "repository", "model", "method", "docs")
                assert res.title != ""

    assert has_linked_resource is True


# -------------------------------------------------------------------------
# 5. REST API ENDPOINTS TESTS
# -------------------------------------------------------------------------

def test_api_explainability_endpoint():
    # Create and persist a run
    doc = make_test_doc("p_api_1", "Paper for API Explanation")
    cand = make_test_candidate(doc)
    model = FormalUtilityModel()
    _, run_rec = model.evaluate(candidates=[cand], persist=True)

    # 1. GET /api/v1/intelligence/explain/{run_id}
    res_get = client.get(f"/api/v1/intelligence/explain/{run_rec.run_id}")
    assert res_get.status_code == 200
    explanations = res_get.json()
    assert len(explanations) == 1
    assert explanations[0]["document_id"] == "p_api_1"
    assert "relevance" in explanations[0]["components"]

    # 2. POST /api/v1/intelligence/explain
    payload = {"run_id": run_rec.run_id, "document_id": "p_api_1"}
    res_post = client.post("/api/v1/intelligence/explain", json=payload)
    assert res_post.status_code == 200
    assert len(res_post.json()) == 1


def test_api_buildability_endpoint():
    doc_data = make_test_doc("p_bld_api", "Buildability API Test Paper").model_dump()
    payload = {"documents": [doc_data]}

    res = client.post("/api/v1/intelligence/buildability", json=payload)
    assert res.status_code == 200
    reports = res.json()
    assert len(reports) == 1
    assert reports[0]["document_id"] == "p_bld_api"
    assert len(reports[0]["dimensions"]) == 9


def test_api_graph_endpoint():
    doc_data = make_test_doc("p_kg_api", "Graph API Test Paper").model_dump()
    payload = {"documents": [doc_data]}

    res = client.post("/api/v1/intelligence/graph", json=payload)
    assert res.status_code == 200
    graph = res.json()
    assert "nodes" in graph
    assert "edges" in graph
    assert graph["total_nodes"] > 0


def test_api_roadmap_endpoint():
    doc_data = make_test_doc("p_rdm_api", "Roadmap API Test Paper").model_dump()
    payload = {"query": "deep learning optimization", "documents": [doc_data]}

    res = client.post("/api/v1/intelligence/roadmap", json=payload)
    assert res.status_code == 200
    roadmap = res.json()
    assert len(roadmap["stages"]) == 7
    assert roadmap["target_query"] == "deep learning optimization"


def test_api_intelligence_report_endpoint():
    doc_data = make_test_doc("p_full_api", "Full Report API Test Paper").model_dump()
    payload = {"query": "transformer systems", "documents": [doc_data]}

    res = client.post("/api/v1/intelligence/report", json=payload)
    assert res.status_code == 200
    report = res.json()
    assert "buildability_reports" in report
    assert "knowledge_graph" in report
    assert "roadmap" in report
    assert len(report["roadmap"]["stages"]) == 7


def test_api_search_with_intelligence_flag(monkeypatch):
    """Verify search endpoint includes intelligence when include_intelligence=True."""
    mock_papers = [make_test_doc("p_srch_1", "Search Intelligence Paper")]

    async def mock_retrieve(self, query, documents, candidate_limit):
        from app.schemas.retrieval import RetrievalMetadata, RetrievedCandidate
        return [
            RetrievedCandidate(
                document=d,
                retrieval_score=0.9,
                retrieval_rank=i + 1,
                metadata=RetrievalMetadata(
                    bm25_rank=i+1, bm25_score=5.0,
                    sbert_rank=i+1, sbert_score=0.8,
                    rrf_rank=i+1, rrf_score=0.03,
                )
            )
            for i, d in enumerate(mock_papers)
        ]

    from app.retrieval.pipeline import HybridRetrievalPipeline
    monkeypatch.setattr(HybridRetrievalPipeline, "retrieve", mock_retrieve)

    payload = {
        "query": "transformer intelligence",
        "limit": 1,
        "include_intelligence": True,
    }

    res = client.post("/api/v1/search", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["intelligence"] is not None
    assert "buildability_reports" in data["intelligence"]
    assert "knowledge_graph" in data["intelligence"]
    assert "roadmap" in data["intelligence"]
