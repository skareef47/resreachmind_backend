"""
Comprehensive tests for Research Architecture Generation in ResearchMind AI.

Validates:
1. Architecture generated from selected papers
2. Evidence attached to every component
3. Unknown values are not fabricated (clearly marked with EvidenceStatus.UNKNOWN)
4. Selected paper changes architecture (different papers yield different architecture components)
5. Constraints affect architecture (edge/latency constraints change inference engine, hardware, and deployment)
6. Multi-paper compatibility and conflict detection (license, hardware, model synergy)
7. POST /api/v1/architecture works
8. GET /api/v1/architecture works
9. Frontend response structure compatibility
"""

import pytest
from fastapi.testclient import TestClient

from app.intelligence.architecture import ArchitectureGenerator
from app.intelligence.models import (
    ArchitectureComponent,
    EvidenceStatus,
    ResearchArchitectureResponse,
)
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
    )


# -----------------------------------------------------------------------------
# 1. ARCHITECTURE GENERATION & EVIDENCE ATTACHMENT
# -----------------------------------------------------------------------------

def test_architecture_generated_from_selected_papers():
    """Verify complete architecture graph is synthesized from candidate paper."""
    doc = make_test_doc(
        "res_1",
        "Deep Residual Learning for Image Recognition",
        abstract="Evaluated on ImageNet 2012 classification dataset using ResNet50 model with SGD optimizer.",
        datasets=["ImageNet"],
        models=["ResNet50"],
        methods=["Residual Connections", "Batch Normalization"],
        code_url="https://github.com/org/resnet",
    )

    generator = ArchitectureGenerator()
    resp = generator.generate_architecture([doc])

    assert isinstance(resp, ResearchArchitectureResponse)
    assert resp.architecture_id.startswith("arch_")
    assert resp.selected_paper_ids == ["doc_res_1"]

    arch = resp.architecture
    assert len(arch.components) >= 8
    assert len(arch.connections) >= 7
    assert len(arch.hardware) >= 2
    assert len(arch.dependencies) >= 3

    # Check component stages exist
    comp_names = {c.component for c in arch.components}
    expected_stages = {"Dataset", "Preprocessing", "Feature Extraction", "Model", "Training", "Evaluation", "Inference", "Backend/API", "Frontend", "Deployment"}
    assert expected_stages.issubset(comp_names)


def test_evidence_is_attached_to_all_components():
    """Every architecture component must contain evidence, confidence, and status."""
    doc = make_test_doc(
        "ev_1",
        "Transformer Language Models with Byte-Pair Encoding",
        abstract="We train Transformer with Byte-Pair Encoding on WikiText-103 using AdamW optimizer and evaluate Perplexity.",
        datasets=["WikiText-103"],
        models=["Transformer"],
        methods=["Multi-Head Attention"],
        code_url="https://github.com/org/transformer",
    )

    generator = ArchitectureGenerator()
    resp = generator.generate_architecture([doc])

    for comp in resp.architecture.components:
        assert isinstance(comp, ArchitectureComponent)
        assert comp.component != ""
        assert comp.value != ""
        assert comp.source_paper != ""
        assert comp.evidence != "", f"Evidence missing for component {comp.component}"
        assert 0.0 <= comp.confidence <= 1.0
        assert comp.status in (EvidenceStatus.VERIFIED, EvidenceStatus.INFERRED, EvidenceStatus.UNKNOWN)

    # Check hardware evidence
    for hw in resp.architecture.hardware:
        assert hw.category != ""
        assert hw.requirement != ""
        assert hw.evidence != ""

    # Check dependency evidence
    for dep in resp.architecture.dependencies:
        assert dep.name != ""
        assert dep.evidence != ""


# -----------------------------------------------------------------------------
# 2. NO FABRICATION OF UNKNOWN VALUES
# -----------------------------------------------------------------------------

def test_unknown_values_not_fabricated():
    """Missing or theoretical implementation details must be flagged as UNKNOWN without fake facts."""
    doc_theoretical = make_test_doc(
        "theo_1",
        "Pure Mathematical Formulations on Convex Subspaces",
        abstract="We prove theorems on Hilbert spaces without empirical datasets or numerical neural baselines.",
        datasets=[],
        models=[],
        methods=[],
        code_url=None,
    )

    generator = ArchitectureGenerator()
    resp = generator.generate_architecture([doc_theoretical])

    dataset_comp = next((c for c in resp.architecture.components if c.component == "Dataset"), None)
    assert dataset_comp is not None
    # Must be marked UNKNOWN or clearly noted lack of empirical dataset
    assert dataset_comp.status == EvidenceStatus.UNKNOWN or "Unknown" in dataset_comp.value
    assert "No public benchmark" in dataset_comp.evidence or "not identified" in dataset_comp.evidence


# -----------------------------------------------------------------------------
# 3. SELECTED PAPER CHANGES ARCHITECTURE
# -----------------------------------------------------------------------------

def test_selected_paper_changes_architecture():
    """Different papers yield distinct, grounded architecture components."""
    doc_nlp = make_test_doc(
        "nlp_1",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        abstract="Pre-trained on BooksCorpus and English Wikipedia with Masked Language Model on Transformer backbone.",
        datasets=["BooksCorpus"],
        models=["BERT"],
        methods=["Masked Language Modeling"],
    )

    doc_cv = make_test_doc(
        "cv_1",
        "Vision Transformer on ImageNet",
        abstract="We evaluate Vision Transformer (ViT) on ImageNet benchmark with Random Augmentation.",
        datasets=["ImageNet"],
        models=["ViT"],
        methods=["Patch Projection"],
    )

    generator = ArchitectureGenerator()
    arch_nlp = generator.generate_architecture([doc_nlp])
    arch_cv = generator.generate_architecture([doc_cv])

    # Model values must differ
    model_nlp = next(c.value for c in arch_nlp.architecture.components if c.component == "Model")
    model_cv = next(c.value for c in arch_cv.architecture.components if c.component == "Model")
    assert model_nlp != model_cv
    assert "BERT" in model_nlp
    assert "ViT" in model_cv

    # Dataset values must differ
    ds_nlp = next(c.value for c in arch_nlp.architecture.components if c.component == "Dataset")
    ds_cv = next(c.value for c in arch_cv.architecture.components if c.component == "Dataset")
    assert ds_nlp != ds_cv
    assert "BooksCorpus" in ds_nlp
    assert "ImageNet" in ds_cv


# -----------------------------------------------------------------------------
# 4. CONSTRAINTS AFFECT ARCHITECTURE
# -----------------------------------------------------------------------------

def test_constraints_affect_architecture():
    """User constraints (e.g. edge environment, tight latency, low VRAM) alter the architecture."""
    doc = make_test_doc(
        "c_1",
        "Heavy Vision Transformer",
        datasets=["ImageNet"],
        models=["ViT-Large"],
    )

    generator = ArchitectureGenerator()

    # Cloud constraint (default)
    arch_cloud = generator.generate_architecture(
        [doc], constraints={"target_environment": "cloud", "max_gpu_memory_gb": 40}
    )
    infer_cloud = next(c.value for c in arch_cloud.architecture.components if c.component == "Inference")
    deploy_cloud = next(c.value for c in arch_cloud.architecture.components if c.component == "Deployment")

    # Edge constraint
    arch_edge = generator.generate_architecture(
        [doc], constraints={"target_environment": "edge", "latency_budget_ms": 30, "max_gpu_memory_gb": 8}
    )
    infer_edge = next(c.value for c in arch_edge.architecture.components if c.component == "Inference")
    deploy_edge = next(c.value for c in arch_edge.architecture.components if c.component == "Deployment")

    assert infer_cloud != infer_edge
    assert "ONNX Runtime" in infer_edge or "Quantization" in infer_edge
    assert "ARM64" in deploy_edge or "Embedded" in deploy_edge or "edge" in deploy_edge.lower()

    # Hardware VRAM budget must reflect constraint
    hw_edge_gpu = next(h for h in arch_edge.architecture.hardware if "GPU" in h.category)
    assert "<= 8GB" in hw_edge_gpu.requirement


# -----------------------------------------------------------------------------
# 5. MULTI-PAPER COMPATIBILITY AND CONFLICTS
# -----------------------------------------------------------------------------

def test_multi_paper_compatibility_and_conflicts():
    """Detect synergies and licensing/compute conflicts across multiple selected papers."""
    doc_open = make_test_doc(
        "p_open",
        "Open SOTA Transformer",
        datasets=["GLUE"],
        models=["Transformer-Small"],
        license_name="MIT",
    )
    doc_gpl = make_test_doc(
        "p_gpl",
        "Copyleft GNN Model",
        datasets=["Cora"],
        models=["GNN"],
        license_name="GPL-3.0",
    )
    doc_prop = make_test_doc(
        "p_prop",
        "Proprietary Industrial Model",
        abstract="Trained on a cluster of 1024 TPUs with proprietary confidential customer data.",
        license_name="Proprietary",
    )

    generator = ArchitectureGenerator()

    # Compatible pair: MIT + GPL (valid in open source pipeline)
    resp_open = generator.generate_architecture([doc_open, doc_gpl])
    assert len(resp_open.compatibility.compatible_points) > 0

    # Conflicting pair: GPL + Proprietary assets
    resp_conflict = generator.generate_architecture([doc_gpl, doc_prop])
    assert len(resp_conflict.compatibility.conflicts) > 0
    assert any("License incompatibility" in c or "Proprietary" in c for c in resp_conflict.compatibility.conflicts)
    assert not resp_conflict.compatibility.is_compatible


# -----------------------------------------------------------------------------
# 6. API ENDPOINTS (POST and GET)
# -----------------------------------------------------------------------------

def test_api_architecture_post_with_documents():
    """POST /api/v1/architecture generates architecture from provided documents."""
    client = TestClient(app)
    doc = make_test_doc(
        "api_p1",
        "Fast Self-Attention Mechanism",
        abstract="Evaluated on GLUE benchmark with Transformer model using AdamW.",
        datasets=["GLUE"],
        models=["Transformer"],
    )

    payload = {
        "documents": [doc.model_dump()],
        "constraints": {
            "target_environment": "cloud",
            "preferred_framework": "PyTorch",
        },
    }

    resp = client.post("/api/v1/architecture", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert "architecture_id" in data
    assert "architecture" in data
    assert "components" in data["architecture"]
    assert "connections" in data["architecture"]
    assert "hardware" in data["architecture"]
    assert "dependencies" in data["architecture"]
    assert "evidence" in data
    assert "compatibility" in data

    components = data["architecture"]["components"]
    assert any(c["component"] == "Model" and "Transformer" in c["value"] for c in components)
    assert any(c["component"] == "Dataset" and "GLUE" in c["value"] for c in components)


def test_api_architecture_get_endpoint():
    """GET /api/v1/architecture?query=... retrieves live papers and returns architecture."""
    client = TestClient(app)
    resp = client.get("/api/v1/architecture?query=transformer+attention&target_environment=cloud")
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert "architecture" in data
    assert len(data["architecture"]["components"]) >= 8
    assert len(data["architecture"]["connections"]) >= 7


# -----------------------------------------------------------------------------
# 7. FRONTEND RENDERS ARCHITECTURE RESPONSE
# -----------------------------------------------------------------------------

def test_frontend_consumes_architecture_response():
    """Ensure response schema contains all keys expected by frontend UI."""
    doc = make_test_doc(
        "fe_1",
        "Generative Models for Speech Synthesis",
        datasets=["LibriSpeech"],
        models=["WaveNet"],
        methods=["Dilated Convolutions"],
    )

    generator = ArchitectureGenerator()
    resp = generator.generate_architecture([doc])
    dump = resp.model_dump()

    # Front-end required attributes
    assert "architecture" in dump
    assert "components" in dump["architecture"]
    assert "connections" in dump["architecture"]
    assert "hardware" in dump["architecture"]
    assert "dependencies" in dump["architecture"]
    assert "compatibility" in dump
    assert "evidence" in dump

    # Check connection structure
    for conn in dump["architecture"]["connections"]:
        assert "source" in conn
        assert "target" in conn
        assert "description" in conn
        assert "data_flow" in conn

    # Check component structure
    for comp in dump["architecture"]["components"]:
        assert "component" in comp
        assert "value" in comp
        assert "evidence" in comp
        assert "confidence" in comp
        assert "status" in comp


def test_end_to_end_search_to_architecture_pipeline():
    """Simulate real search results being passed into POST /api/v1/architecture."""
    client = TestClient(app)

    # 1. Retrieve search results
    search_resp = client.post("/api/v1/search", json={
        "query": "resnet convolutional neural network",
        "limit": 2,
        "sources": ["openalex"],
        "optimize": False,
    })
    assert search_resp.status_code == 200, search_resp.text
    search_data = search_resp.json()
    assert "results" in search_data
    docs = search_data["results"]
    assert len(docs) > 0

    # 2. Extract returned document IDs and payloads
    doc_ids = [d["internal_id"] for d in docs]

    # 3. Request concrete architecture for these documents with edge constraint
    arch_resp = client.post("/api/v1/architecture", json={
        "paper_ids": doc_ids,
        "documents": docs,
        "constraints": {
            "target_environment": "edge",
            "max_gpu_memory_gb": 8,
        },
    })
    assert arch_resp.status_code == 200, arch_resp.text
    arch_data = arch_resp.json()

    assert arch_data["selected_paper_ids"] == doc_ids
    assert len(arch_data["architecture"]["components"]) == 10
    assert arch_data["constraints_applied"]["target_environment"] == "edge"
    assert arch_data["constraints_applied"]["max_gpu_memory_gb"] == 8

    # Verify edge constraints shifted inference engine to ONNX / INT8
    inference_comp = next(c for c in arch_data["architecture"]["components"] if c["component"] == "Inference")
    assert "ONNX" in inference_comp["value"] or "Quantization" in inference_comp["value"]

    # Verify evidence items are attached
    assert len(arch_data["evidence"]) >= 10
    for ev in arch_data["evidence"]:
        assert ev["evidence"] != ""
        assert ev["status"] in ["verified", "inferred", "unknown"]

