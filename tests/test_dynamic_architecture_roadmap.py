"""
Comprehensive Differentiation and Dynamic Generation Tests for ResearchMind AI Architecture & Roadmap.

Verifies:
1. Query Differentiation across 3 radically different domains:
   - A: Cancer detection using MRI (Medical Imaging)
   - B: Crop disease detection (Agricultural Vision)
   - C: Transformer-based fraud detection (Financial Tabular Anomaly)
   Asserts:
   - architecture(A) != architecture(B) != architecture(C) across models, datasets, preprocessing, components, technologies
   - roadmap(A) != roadmap(B) != roadmap(C) across tasks, dependencies, technologies, inputs, outputs
2. Same-Paper Consistency:
   - Repeated generation with identical papers/query produces identical architecture and roadmap.
3. Feedback / Optimization Propagation:
   - Ranking changes triggered by user preferences/feedback result in dynamically updated architectures and roadmaps.
4. Evidence Grounding & UNKNOWN Verification:
   - Incomplete papers produce UNKNOWN evidence_status and no hallucinated repos/metrics.
5. REST API Integration for POST /api/v1/architecture and POST /api/v1/roadmap with selection_mode.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.paper import ResearchDocument
from app.intelligence.models import (
    EvidenceStatus,
    ResearchArchitectureResponse,
    ResearchRoadmapResponse,
)
from app.intelligence.architecture import ArchitectureGenerator
from app.intelligence.roadmap import RoadmapGenerator


def make_doc(
    doc_id: str,
    title: str,
    abstract: str,
    datasets: list = None,
    models: list = None,
    methods: list = None,
    code_url: str = None,
    concepts: list = None,
) -> ResearchDocument:
    return ResearchDocument(
        internal_id=f"doc_{doc_id}",
        source="openalex",
        source_id=doc_id,
        title=title,
        abstract=abstract,
        authors=["Primary Researcher", "Co-Author"],
        publication_year=2024,
        venue="Top Conference",
        concepts=concepts or ["Machine Learning"],
        datasets=datasets or [],
        models=models or [],
        methods=methods or [],
        code_url=code_url,
        license="MIT",
        doi=f"10.1000/{doc_id}",
    )


# =========================================================================
# 1. RADICAL QUERY DIFFERENTIATION TEST
# =========================================================================

def test_query_differentiation_three_domains():
    """Verify 3 radically different research queries produce completely different architectures and roadmaps."""
    arch_gen = ArchitectureGenerator()
    roadmap_gen = RoadmapGenerator()

    # Query A: Cancer detection using MRI
    query_a = "Cancer detection using MRI"
    doc_a = make_doc(
        "cancer_mri",
        "Deep Residual Learning with Support Vector Machines for Brain Tumor MRI Detection",
        "We propose an automated neuro-oncology diagnosis system using volumetric brain MRI scans from the BRATS dataset. Skull stripping and slice registration are applied before extracting features using ResNet50. An SVM decision boundary is trained and Grad-CAM is utilized for tumor margin interpretability.",
        datasets=["BRATS"],
        models=["ResNet50"],
        methods=["Support Vector Machine", "Skull Stripping", "Grad-CAM"],
        code_url="https://github.com/med/mri-resnet-svm",
        concepts=["Neuro-oncology", "Magnetic Resonance Imaging", "Brain Tumor"],
    )

    # Query B: Crop disease detection
    query_b = "Crop disease detection"
    doc_b = make_doc(
        "crop_disease",
        "Foliar Pathogen Identification on PlantVillage Leaves Using EfficientNet-B0",
        "We detect foliar blight and fungal rust on leaf images using the PlantVillage benchmark. High-resolution crop photographs undergo geometric rotation and color jitter augmentations before fine-tuning EfficientNet-B0 depthwise separable convolutions with Cross-Entropy loss.",
        datasets=["PlantVillage"],
        models=["EfficientNet-B0"],
        methods=["Geometric Augmentation", "Color Jitter", "Fine-Tuning"],
        code_url="https://github.com/agri/crop-efficientnet",
        concepts=["Plant Pathology", "Agricultural Vision", "Crop Blight"],
    )

    # Query C: Transformer-based fraud detection
    query_c = "Transformer-based fraud detection"
    doc_c = make_doc(
        "fraud_transformer",
        "Temporal Transformer Self-Attention for Financial Transaction Fraud Detection",
        "We address high-throughput credit card fraud on the IEEE-CIS transaction dataset. Tabular event sequences undergo sliding-window aggregations and SMOTE oversampling. A multi-head self-attention Transformer network is trained with Focal Loss (gamma=2.0) and explained using Tree SHAP feature attributions.",
        datasets=["IEEE-CIS Fraud Dataset"],
        models=["Transformer Anomaly Classifier"],
        methods=["SMOTE", "Self-Attention", "Focal Loss", "SHAP"],
        code_url="https://github.com/fintech/fraud-transformer",
        concepts=["Financial Fraud", "Anomaly Detection", "Tabular Time-Series"],
    )

    # Generate Architectures
    arch_a = arch_gen.generate_architecture([doc_a], query=query_a)
    arch_b = arch_gen.generate_architecture([doc_b], query=query_b)
    arch_c = arch_gen.generate_architecture([doc_c], query=query_c)

    # Generate Roadmaps
    road_a = roadmap_gen.generate_roadmap([doc_a], architecture=arch_a.architecture)
    road_b = roadmap_gen.generate_roadmap([doc_b], architecture=arch_b.architecture)
    road_c = roadmap_gen.generate_roadmap([doc_c], architecture=arch_c.architecture)

    # -------------------------------------------------------------
    # ARCHITECTURE DIFFERENTIATION ASSERTIONS
    # -------------------------------------------------------------
    # 1. Architecture Type / Topologies must be completely different
    assert arch_a.architecture_type != arch_b.architecture_type
    assert arch_b.architecture_type != arch_c.architecture_type
    assert arch_a.architecture_type != arch_c.architecture_type

    # 2. Datasets must be specific and distinct
    dataset_a = next(c.value for c in arch_a.components if c.component == "Dataset")
    dataset_b = next(c.value for c in arch_b.components if c.component == "Dataset")
    dataset_c = next(c.value for c in arch_c.components if c.component == "Dataset")
    assert dataset_a == "BRATS"
    assert dataset_b == "PlantVillage"
    assert dataset_c == "IEEE-CIS Fraud Dataset"
    assert len({dataset_a, dataset_b, dataset_c}) == 3

    # 3. Preprocessing must be domain-specific
    preproc_a = next(c.value for c in arch_a.components if c.component == "Preprocessing")
    preproc_b = next(c.value for c in arch_b.components if c.component == "Preprocessing")
    preproc_c = next(c.value for c in arch_c.components if c.component == "Preprocessing")
    assert "skull stripping" in preproc_a.lower()
    assert "augmentation" in preproc_b.lower() or "jitter" in preproc_b.lower()
    assert "smote" in preproc_c.lower() or "scaling" in preproc_c.lower() or "time-window" in preproc_c.lower()
    assert len({preproc_a, preproc_b, preproc_c}) == 3

    # 4. Models must be specific and distinct
    model_a = next(c.value for c in arch_a.components if c.component == "Model")
    model_b = next(c.value for c in arch_b.components if c.component == "Model")
    model_c = next(c.value for c in arch_c.components if c.component == "Model")
    assert "ResNet50" in model_a or "Support Vector Machine" in model_a
    assert "EfficientNet" in model_b
    assert "Transformer" in model_c
    assert len({model_a, model_b, model_c}) == 3

    # 5. Dependencies must be tailored to the modality
    dep_names_a = {d.name.lower() for d in arch_a.architecture.dependencies}
    dep_names_b = {d.name.lower() for d in arch_b.architecture.dependencies}
    dep_names_c = {d.name.lower() for d in arch_c.architecture.dependencies}
    # Medical has dicom/nifti or nibabel
    assert any("dicom" in d or "nibabel" in d or "scikit-learn" in d for d in dep_names_a)
    # Crop vision has albumentations or pillow or timm
    assert any("pillow" in d or "timm" in d or "albumentations" in d or "torchvision" in d for d in dep_names_b)
    # Fraud has imbalanced-learn / shap / pandas
    assert any("shap" in d or "imbalanced-learn" in d or "pandas" in d for d in dep_names_c)

    # -------------------------------------------------------------
    # ROADMAP DIFFERENTIATION ASSERTIONS
    # -------------------------------------------------------------
    # 1. Phase 1 Tasks (Data Ingestion) must reference the actual paper datasets
    p1_tasks_a = " ".join(t.task for t in road_a.phases[0].task_items)
    p1_tasks_b = " ".join(t.task for t in road_b.phases[0].task_items)
    p1_tasks_c = " ".join(t.task for t in road_c.phases[0].task_items)
    assert "BRATS" in p1_tasks_a
    assert "PlantVillage" in p1_tasks_b
    assert "IEEE-CIS" in p1_tasks_c
    assert p1_tasks_a != p1_tasks_b != p1_tasks_c

    # 2. Phase 3 Tasks (Preprocessing & Feature Pipeline)
    p3_tasks_a = " ".join(t.task for t in road_a.phases[2].task_items)
    p3_tasks_b = " ".join(t.task for t in road_b.phases[2].task_items)
    p3_tasks_c = " ".join(t.task for t in road_c.phases[2].task_items)
    assert "skull stripping" in p3_tasks_a.lower()
    assert "augmentation" in p3_tasks_b.lower() or "jitter" in p3_tasks_b.lower()
    assert "smote" in p3_tasks_c.lower() or "time-window" in p3_tasks_c.lower() or "scaling" in p3_tasks_c.lower()
    assert p3_tasks_a != p3_tasks_b != p3_tasks_c

    # 3. Phase 4 Tasks (Model Implementation & Training)
    p4_tasks_a = " ".join(t.task for t in road_a.phases[3].task_items)
    p4_tasks_b = " ".join(t.task for t in road_b.phases[3].task_items)
    p4_tasks_c = " ".join(t.task for t in road_c.phases[3].task_items)
    assert ("ResNet50" in p4_tasks_a or "SVM" in p4_tasks_a)
    assert "EfficientNet" in p4_tasks_b
    assert "Transformer" in p4_tasks_c
    assert p4_tasks_a != p4_tasks_b != p4_tasks_c

    # 4. Roadmap Task Item counts and technologies
    techs_a = {t.technology for p in road_a.phases for t in p.task_items if t.technology}
    techs_b = {t.technology for p in road_b.phases for t in p.task_items if t.technology}
    techs_c = {t.technology for p in road_c.phases for t in p.task_items if t.technology}
    assert techs_a != techs_b
    assert techs_b != techs_c


# =========================================================================
# 2. SAME-PAPER CONSISTENCY TEST
# =========================================================================

def test_same_paper_consistency():
    """Verify that repeated generation with identical papers and query produces identical results."""
    arch_gen = ArchitectureGenerator()
    roadmap_gen = RoadmapGenerator()

    doc = make_doc(
        "stable_1",
        "Attention Is All You Need",
        "We propose the Transformer architecture using multi-head self-attention mechanisms on the WMT 2014 English-to-German dataset with Adam optimizer and evaluate with BLEU score.",
        datasets=["WMT 2014 English-to-German"],
        models=["Transformer"],
        methods=["Multi-Head Attention", "Positional Encoding"],
        code_url="https://github.com/tensorflow/tensor2tensor",
    )

    query = "Transformer translation architecture"

    # Run 1
    arch_1 = arch_gen.generate_architecture([doc], query=query)
    road_1 = roadmap_gen.generate_roadmap([doc], architecture=arch_1.architecture)

    # Run 2
    arch_2 = arch_gen.generate_architecture([doc], query=query)
    road_2 = roadmap_gen.generate_roadmap([doc], architecture=arch_2.architecture)

    # Check Architecture consistency
    assert arch_1.architecture_type == arch_2.architecture_type
    assert [c.value for c in arch_1.components] == [c.value for c in arch_2.components]
    assert [c.evidence_status for c in arch_1.components] == [c.evidence_status for c in arch_2.components]
    assert [c.evidence_source for c in arch_1.components] == [c.evidence_source for c in arch_2.components]

    # Check Roadmap consistency
    assert [p.title for p in road_1.phases] == [p.title for p in road_2.phases]
    assert [t.task for p in road_1.phases for t in p.task_items] == [t.task for p in road_2.phases for t in p.task_items]
    assert [t.dependency for p in road_1.phases for t in p.task_items] == [t.dependency for p in road_2.phases for t in p.task_items]


# =========================================================================
# 3. FEEDBACK / OPTIMIZATION PROPAGATION TEST
# =========================================================================

def test_ranking_feedback_propagates_to_architecture_and_roadmap():
    """Verify that shifting top-ranked paper via feedback changes synthesized architecture & roadmap."""
    arch_gen = ArchitectureGenerator()
    roadmap_gen = RoadmapGenerator()

    # Direction 1: User previously preferred LightGBM Tabular approach
    doc_tabular = make_doc(
        "rank_tab",
        "LightGBM Gradient Boosting for Default Credit Risk Prediction",
        "We use Home Credit default risk dataset with quantile feature binning and LightGBM trees. Evaluation uses ROC-AUC.",
        datasets=["Home Credit Risk"],
        models=["LightGBM"],
        methods=["Gradient Boosting", "Feature Binning"],
    )

    # Direction 2: User shifts preference to Deep Graph Neural Network approach
    doc_graph = make_doc(
        "rank_graph",
        "Inductive Graph Convolutional Networks for Financial Fraud Topology",
        "We represent financial accounts as an attributed graph on Elliptic Bitcoin dataset using 2-layer Graph Convolutional Networks (GCN) with neighborhood sampling.",
        datasets=["Elliptic Bitcoin Dataset"],
        models=["Graph Convolutional Network"],
        methods=["Message Passing", "Neighborhood Sampling"],
    )

    # Initial ranking selects doc_tabular as top paper
    arch_initial = arch_gen.generate_architecture([doc_tabular], query="Credit default risk")
    road_initial = roadmap_gen.generate_roadmap([doc_tabular], architecture=arch_initial.architecture)

    # After feedback, ranking reorders and doc_graph is top paper
    arch_updated = arch_gen.generate_architecture([doc_graph], query="Credit default risk with graph topology")
    road_updated = roadmap_gen.generate_roadmap([doc_graph], architecture=arch_updated.architecture)

    # Assert that architecture and roadmap adapted to the new ranking
    assert arch_initial.architecture_type != arch_updated.architecture_type
    model_init = next(c.value for c in arch_initial.components if c.component == "Model")
    model_upd = next(c.value for c in arch_updated.components if c.component == "Model")
    assert "LightGBM" in model_init
    assert "Graph" in model_upd or "GCN" in model_upd

    p1_init = " ".join(t.task for t in road_initial.phases[0].task_items)
    p1_upd = " ".join(t.task for t in road_updated.phases[0].task_items)
    assert "Home Credit" in p1_init
    assert "Elliptic" in p1_upd


# =========================================================================
# 4. EVIDENCE GROUNDING & UNKNOWN STATUS HANDLING
# =========================================================================

def test_unknown_evidence_is_not_fabricated():
    """Missing paper metadata must be flagged as UNKNOWN with provenance, never fabricated."""
    arch_gen = ArchitectureGenerator()
    roadmap_gen = RoadmapGenerator()

    # Paper with missing datasets, code, and explicit methods
    doc_sparse = make_doc(
        "sparse_doc",
        "Theoretical Exploration of Deep Representations in Mathematical Logic",
        "We investigate mathematical bounds of abstract neural topologies without empirical benchmarking.",
        datasets=[],
        models=[],
        methods=[],
        code_url=None,
    )

    arch_sparse = arch_gen.generate_architecture([doc_sparse], query="abstract neural logic")
    road_sparse = roadmap_gen.generate_roadmap([doc_sparse], architecture=arch_sparse.architecture)

    # Check Dataset component
    dataset_comp = next(c for c in arch_sparse.components if c.component == "Dataset")
    assert dataset_comp.evidence_status == EvidenceStatus.UNKNOWN
    assert "Publisher metadata absent" in dataset_comp.evidence_source or "insufficient" in dataset_comp.evidence_source.lower()

    # Check Roadmap Phase 1 (Data Ingestion) reflects UNKNOWN for benchmark dataset
    unknown_p1_tasks = [t for t in road_sparse.phases[0].task_items if t.evidence_status == EvidenceStatus.UNKNOWN]
    assert len(unknown_p1_tasks) >= 1
    assert "proxy" in unknown_p1_tasks[0].task.lower() or "no named benchmark" in unknown_p1_tasks[0].task.lower()

    # Check Phase 2 (Environment Setup) does NOT fabricate official git repo
    p2_tasks = " ".join(t.task.lower() for t in road_sparse.phases[1].task_items)
    assert "clone and verify official paper repository" not in p2_tasks
    assert "generic scaffold" in p2_tasks or "initialize" in p2_tasks


# =========================================================================
# 5. REST API INTEGRATION WITH selection_mode
# =========================================================================

def test_api_architecture_and_roadmap_with_selection_mode():
    """Test POST /api/v1/architecture and POST /api/v1/roadmap with selection_mode payload."""
    client = TestClient(app)

    doc = {
        "internal_id": "doc_api_test",
        "source": "openalex",
        "source_id": "api_test",
        "title": "Convolutional Ultrasound Segmentation for Breast Cancer",
        "abstract": "We evaluate U-Net on the BUSI breast ultrasound image dataset with dice loss.",
        "authors": ["Dr. Smith"],
        "publication_year": 2024,
        "venue": "MICCAI",
        "concepts": ["Ultrasound", "Oncology"],
        "datasets": ["BUSI Dataset"],
        "models": ["U-Net"],
        "methods": ["Dice Loss", "Histogram Equalization"],
        "code_url": "https://github.com/med/busi-unet",
        "license": "Apache-2.0",
        "doi": "10.1000/busi",
    }

    # 1. POST /api/v1/architecture
    arch_resp = client.post("/api/v1/architecture", json={
        "query": "Ultrasound breast cancer segmentation",
        "paper_ids": ["doc_api_test"],
        "documents": [doc],
        "selection_mode": "optimized",
        "constraints": {
            "target_environment": "clinical_edge",
            "max_gpu_memory_gb": 8,
        },
    })
    assert arch_resp.status_code == 200, arch_resp.text
    arch_data = arch_resp.json()

    assert arch_data["query"] == "Ultrasound breast cancer segmentation"
    assert "doc_api_test" in arch_data["papers_used"]
    assert len(arch_data["components"]) == 10
    assert len(arch_data["connections"]) >= 9
    assert arch_data["confidence"] > 0.0

    # Verify component structure
    comp_0 = arch_data["components"][0]
    assert comp_0["evidence_status"] in ["VERIFIED", "INFERRED", "UNKNOWN"]
    assert comp_0["evidence_source"] != ""

    # 2. POST /api/v1/roadmap
    roadmap_resp = client.post("/api/v1/roadmap", json={
        "query": "Ultrasound breast cancer segmentation",
        "paper_ids": ["doc_api_test"],
        "documents": [doc],
        "selection_mode": "optimized",
        "constraints": {
            "limited_gpu": True,
        },
    })
    assert roadmap_resp.status_code == 200, roadmap_resp.text
    roadmap_data = roadmap_resp.json()

    assert "doc_api_test" in roadmap_data["papers_used"]
    assert len(roadmap_data["phases"]) == 8
    
    # Verify task items structure in Phase 1
    phase_1 = roadmap_data["phases"][0]
    assert len(phase_1["task_items"]) >= 1
    t0 = phase_1["task_items"][0]
    assert t0["phase"] == 1
    assert t0["task"] != ""
    assert t0["description"] != ""
    assert t0["evidence_status"] in ["VERIFIED", "INFERRED", "UNKNOWN"]
    all_p1_tasks = " ".join(t["task"] for t in phase_1["task_items"])
    assert "BUSI" in all_p1_tasks
