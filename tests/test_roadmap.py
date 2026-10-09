"""
Comprehensive tests for Research Implementation Roadmap Generation in ResearchMind AI.

Validates:
1. Roadmap generated from selected research papers
2. Phases are strictly ordered (1 to 8)
3. Dependencies are valid (DAG, referencing prerequisite phases)
4. Constraints affect roadmap (limited GPU, limited time, public datasets, open-source)
5. Evidence is included in each phase
6. Unknown information remains unknown (no fabricated repos, hardware, datasets)
7. Selected paper changes roadmap output
8. POST /api/v1/roadmap works with documents and constraints
9. GET /api/v1/roadmap works
10. Frontend compatibility and end-to-end integration
"""

import pytest
from fastapi.testclient import TestClient

from app.intelligence.models import (
    ResearchRoadmapResponse,
    RoadmapPhase,
)
from app.intelligence.roadmap import RoadmapGenerator
from app.main import app
from app.schemas.paper import ResearchDocument


def make_test_doc(
    doc_id: str,
    title: str,
    abstract: str = "",
    authors: list = None,
    datasets: list = None,
    models: list = None,
    methods: list = None,
    code_url: str = None,
    license_name: str = "MIT",
    venue: str = "NeurIPS",
    concepts: list = None,
    buildability_breakdown: dict = None,
) -> ResearchDocument:
    return ResearchDocument(
        internal_id=f"doc_{doc_id}",
        source="openalex",
        source_id=doc_id,
        title=title,
        abstract=abstract or f"We present {title} with empirical experiments.",
        authors=authors or ["Alice Researcher", "Bob Engineer"],
        publication_year=2023,
        venue=venue,
        concepts=concepts or ["Machine Learning"],
        datasets=datasets or [],
        models=models or [],
        methods=methods or [],
        code_url=code_url,
        license=license_name,
        doi=f"10.1000/{doc_id}",
        buildability_breakdown=buildability_breakdown or {},
    )


# -----------------------------------------------------------------------------
# 1. ROADMAP GENERATION & ORDERING
# -----------------------------------------------------------------------------

def test_roadmap_generated_from_selected_research():
    """Verify that an 8-phase roadmap is generated from selected research."""
    doc = make_test_doc(
        "res_1",
        "Deep Residual Learning for Image Recognition",
        abstract="Evaluated on ImageNet 2012 classification dataset using ResNet50 model with SGD optimizer.",
        datasets=["ImageNet"],
        models=["ResNet50"],
        methods=["Residual Connections", "Batch Normalization"],
        code_url="https://github.com/org/resnet",
    )

    generator = RoadmapGenerator()
    resp = generator.generate_roadmap([doc])

    assert isinstance(resp, ResearchRoadmapResponse)
    assert resp.roadmap_id.startswith("roadmap_")
    assert resp.selected_paper_ids == ["doc_res_1"]
    assert len(resp.phases) == 8

    # Verify all 8 phases are ordered 1 to 8
    for idx, phase in enumerate(resp.phases, start=1):
        assert isinstance(phase, RoadmapPhase)
        assert phase.phase == idx
        assert phase.title != ""
        assert len(phase.tasks) >= 2
        assert len(phase.inputs) >= 1
        assert len(phase.outputs) >= 1
        assert phase.status == "planned"
        assert phase.estimated_difficulty in ["Low", "Medium", "High", "Very High"]


def test_phases_are_ordered_and_dependencies_valid():
    """Verify dependency DAG: prerequisites must only reference earlier phases."""
    doc = make_test_doc(
        "dep_1",
        "Attention Is All You Need",
        abstract="Evaluated on WMT 2014 English-to-German using Transformer model.",
        datasets=["WMT 2014"],
        models=["Transformer"],
    )

    generator = RoadmapGenerator()
    resp = generator.generate_roadmap([doc])

    phase_titles_seen = set()
    phase_numbers_seen = set()

    for p in resp.phases:
        # Phase 1 must have no prerequisites
        if p.phase == 1:
            assert len(p.dependencies) == 0

        # Any prerequisite must be an already seen phase title
        for dep in p.dependencies:
            assert dep in phase_titles_seen, f"Dependency '{dep}' not satisfied before Phase {p.phase}"

        phase_titles_seen.add(p.title)
        phase_numbers_seen.add(p.phase)

    # All phases 1 through 8 must be present
    assert phase_numbers_seen == set(range(1, 9))

    # Dependency graph edges must point from lower to higher phase
    assert len(resp.dependency_graph) >= 7
    for edge in resp.dependency_graph:
        assert edge.from_phase < edge.to_phase
        assert edge.reason != ""


# -----------------------------------------------------------------------------
# 2. EVIDENCE GROUNDING & BUILDABILITY INFORMATION
# -----------------------------------------------------------------------------

def test_evidence_is_included_and_code_recommended_when_available():
    """When code is available, code integration tasks and verified evidence are attached."""
    doc = make_test_doc(
        "ev_code",
        "Fast and Accurate Text Classification",
        code_url="https://github.com/facebookresearch/fastText",
        datasets=["AG News"],
        models=["fastText"],
    )

    generator = RoadmapGenerator()
    resp = generator.generate_roadmap([doc])

    # Phase 2 should recommend cloning verified repo
    p2 = resp.phases[1]
    assert any("clone and verify official paper repository" in t.lower() for t in p2.tasks)
    assert any("fasttext" in t.lower() for t in p2.tasks)

    # Phase 4 should recommend importing modules from cloned repo
    p4 = resp.phases[3]
    assert any("import" in t.lower() and "fasttext" in t.lower() for t in p4.tasks)

    # Evidence should be attached
    assert len(resp.evidence) >= 1
    repo_ev = next((e for e in resp.evidence if e.get("component") == "Repository"), None)
    assert repo_ev is not None
    assert repo_ev["status"] == "verified"
    assert "https://github.com/facebookresearch/fastText" in repo_ev["value"]


def test_unknown_information_remains_unknown():
    """Missing repositories or hardware must be marked UNKNOWN without fabrication."""
    doc_theoretical = make_test_doc(
        "ev_unknown",
        "Pure Mathematical Analysis of High-Dimensional Manifolds",
        abstract="We present mathematical theorems on Riemannian geometry.",
        code_url=None,
        datasets=[],
        models=[],
    )

    generator = RoadmapGenerator()
    resp = generator.generate_roadmap([doc_theoretical])

    # Phase 1 dataset should be marked unknown
    p1 = resp.phases[0]
    p1_ev = next((e for e in p1.evidence if e.get("component") == "Dataset"), None)
    assert p1_ev is not None
    assert p1_ev["status"] == "unknown"
    assert "proxy" in p1_ev["evidence"].lower() or "not confirmed" in p1_ev["evidence"].lower()

    # Phase 2 repository and hardware should be marked unknown
    p2 = resp.phases[1]
    p2_ev_repo = next((e for e in p2.evidence if e.get("component") == "Repository"), None)
    assert p2_ev_repo is not None
    assert p2_ev_repo["status"] == "unknown"

    p2_ev_hw = next((e for e in p2.evidence if e.get("component") == "Hardware"), None)
    assert p2_ev_hw is not None
    assert p2_ev_hw["status"] == "unknown"


# -----------------------------------------------------------------------------
# 3. CONSTRAINTS AFFECT ROADMAP
# -----------------------------------------------------------------------------

def test_constraints_affect_roadmap():
    """User constraints (limited GPU, limited time, public datasets, open source) alter tasks."""
    doc = make_test_doc(
        "const_1",
        "Language Models are Few-Shot Learners",
        abstract="We train GPT-3 language models on Common Crawl.",
        datasets=["Common Crawl"],
        models=["GPT-3"],
    )

    generator = RoadmapGenerator()

    # 1. Unconstrained
    resp_unconstrained = generator.generate_roadmap([doc], constraints={})
    assert "Full Engineering Implementation" in resp_unconstrained.total_estimated_duration

    # 2. Constrained: limited GPU and limited time
    resp_constrained = generator.generate_roadmap(
        [doc],
        constraints={
            "limited_gpu": True,
            "max_gpu_memory_gb": 8,
            "limited_time": True,
            "public_datasets_only": True,
            "open_source_only": True,
        },
    )

    # Duration should be significantly reduced
    assert "3-5 days" in resp_constrained.total_estimated_duration

    # Phase 2 should have quantization and open-source audit tasks
    p2 = resp_constrained.phases[1]
    assert any("quantization" in t.lower() or "bitsandbytes" in t.lower() for t in p2.tasks)
    assert any("open-source" in t.lower() for t in p2.tasks)

    # Phase 4 should have LoRA / PEFT task
    p4 = resp_constrained.phases[3]
    assert any("lora" in t.lower() or "peft" in t.lower() or "quantized" in t.lower() for t in p4.tasks)

    # Phase 5 should have gradient accumulation task
    p5 = resp_constrained.phases[4]
    assert any("gradient accumulation" in t.lower() or "fine-tuning" in t.lower() for t in p5.tasks)

    # Phase 7 should have INT8 quantization task
    p7 = resp_constrained.phases[6]
    assert any("int8" in t.lower() or "quantization" in t.lower() for t in p7.tasks)


def test_selected_paper_changes_roadmap():
    """Different selected papers produce different roadmap details and models."""
    doc_vision = make_test_doc(
        "p_vis",
        "Vision Transformer for Image Recognition",
        datasets=["ImageNet-21k"],
        models=["ViT-B/16"],
        code_url="https://github.com/google-research/vision_transformer",
    )
    doc_audio = make_test_doc(
        "p_aud",
        "Wav2Vec 2.0: A Framework for Self-Supervised Speech",
        datasets=["LibriSpeech"],
        models=["Wav2Vec2"],
        code_url="https://github.com/facebookresearch/fairseq",
    )

    generator = RoadmapGenerator()
    resp_vis = generator.generate_roadmap([doc_vision])
    resp_aud = generator.generate_roadmap([doc_audio])

    # Vision roadmap references ViT and ImageNet
    assert "ViT" in resp_vis.phases[3].title or any("vit" in t.lower() for t in resp_vis.phases[3].tasks)
    assert any("imagenet" in t.lower() for t in resp_vis.phases[0].tasks)

    # Audio roadmap references Wav2Vec and LibriSpeech
    assert "Wav2Vec" in resp_aud.phases[3].title or any("wav2vec" in t.lower() for t in resp_aud.phases[3].tasks)
    assert any("librispeech" in t.lower() for t in resp_aud.phases[0].tasks)


# -----------------------------------------------------------------------------
# 4. API VERIFICATION
# -----------------------------------------------------------------------------

def test_api_roadmap_post_with_documents():
    """POST /api/v1/roadmap accepts documents and constraints, returning valid roadmap."""
    client = TestClient(app)
    doc = make_test_doc(
        "api_rm_1",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        datasets=["BooksCorpus", "English Wikipedia"],
        models=["BERT-Base"],
        code_url="https://github.com/google-research/bert",
    )

    payload = {
        "paper_ids": [doc.internal_id],
        "documents": [doc.model_dump()],
        "constraints": {
            "limited_gpu": True,
            "max_gpu_memory_gb": 8,
        },
    }

    resp = client.post("/api/v1/roadmap", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert "roadmap_id" in data
    assert "phases" in data
    assert len(data["phases"]) == 8
    assert data["selected_paper_ids"] == [doc.internal_id]
    assert data["constraints_applied"]["limited_gpu"] is True
    assert len(data["dependency_graph"]) >= 7


def test_api_roadmap_get_endpoint():
    """GET /api/v1/roadmap?query=... retrieves live papers and returns roadmap."""
    client = TestClient(app)
    resp = client.get("/api/v1/roadmap?query=bert+transformer&limited_gpu=true")
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert "phases" in data
    assert len(data["phases"]) == 8
    assert data["constraints_applied"]["limited_gpu"] is True


def test_frontend_consumes_roadmap_response():
    """Ensure response schema contains all keys expected by frontend UI."""
    doc = make_test_doc(
        "fe_rm",
        "Graph Convolutional Networks for Semi-Supervised Classification",
        datasets=["Cora", "Citeseer"],
        models=["GCN"],
    )

    generator = RoadmapGenerator()
    resp = generator.generate_roadmap([doc])
    dump = resp.model_dump()

    assert "roadmap_id" in dump
    assert "title" in dump
    assert "phases" in dump
    assert "dependency_graph" in dump
    assert "total_estimated_duration" in dump
    assert "constraints_applied" in dump

    # Check phase fields
    for p in dump["phases"]:
        assert "phase" in p
        assert "title" in p
        assert "description" in p
        assert "tasks" in p
        assert "dependencies" in p
        assert "inputs" in p
        assert "outputs" in p
        assert "estimated_difficulty" in p
        assert "evidence" in p
        assert "status" in p


def test_end_to_end_search_to_roadmap_pipeline():
    """Simulate real search results being passed into POST /api/v1/roadmap."""
    client = TestClient(app)

    # 1. Retrieve search results
    search_resp = client.post("/api/v1/search", json={
        "query": "resnet classification",
        "limit": 2,
        "sources": ["openalex"],
        "optimize": False,
    })
    assert search_resp.status_code == 200, search_resp.text
    search_data = search_resp.json()
    assert "results" in search_data
    docs = search_data["results"]
    assert len(docs) > 0

    doc_ids = [d["internal_id"] for d in docs]

    # 2. Request roadmap with constraints
    roadmap_resp = client.post("/api/v1/roadmap", json={
        "paper_ids": doc_ids,
        "documents": docs,
        "constraints": {
            "limited_gpu": True,
            "max_gpu_memory_gb": 8,
            "limited_time": True,
        },
    })
    assert roadmap_resp.status_code == 200, roadmap_resp.text
    roadmap_data = roadmap_resp.json()

    assert roadmap_data["selected_paper_ids"] == doc_ids
    assert len(roadmap_data["phases"]) == 8
    assert "Fast-Track" in roadmap_data["total_estimated_duration"]
    assert len(roadmap_data["dependency_graph"]) >= 7
