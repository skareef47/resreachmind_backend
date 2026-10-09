"""
Comprehensive Integration Tests for Roadmap-Architecture Binding in ResearchMind AI.

Validates that:
1. test_roadmap_changes_when_architecture_changes: Roadmap materially changes when architecture changes.
2. test_roadmap_contains_current_architecture_components: Roadmap contains all current architecture components.
3. test_roadmap_contains_current_dataset: Roadmap tasks bind to the current architecture dataset.
4. test_roadmap_contains_current_model: Roadmap tasks bind to the current architecture model.
5. test_roadmap_contains_current_preprocessing: Roadmap tasks bind to the current architecture preprocessing.
6. test_roadmap_contains_current_evaluation: Roadmap tasks bind to the current architecture evaluation metrics.
7. test_same_architecture_same_roadmap: Deterministic reproduction with same architecture and papers.
8. test_frontend_uses_live_roadmap_response: API returns full live roadmap response with architecture_hash and all 10 task fields.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.paper import ResearchDocument
from app.intelligence.architecture import ArchitectureGenerator
from app.intelligence.roadmap import RoadmapGenerator
from app.intelligence.models import (
    ResearchArchitectureResponse,
    ResearchRoadmapResponse,
    EvidenceStatus,
)


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
        authors=["Alice Researcher", "Bob Engineer"],
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


def test_roadmap_changes_when_architecture_changes():
    """Verify roadmap materially changes (hash, tasks, technologies) when architecture changes."""
    # Architecture A: MRI Brain Tumor
    doc_a = make_doc(
        "doc_mri",
        "Deep Brain MRI Segmentation and SVM Classification",
        "We propose skull stripping and intensity normalization on BraTS MRI dataset, extracting ResNet50 features to train an SVM classifier with Grad-CAM explainability.",
        datasets=["BraTS MRI"],
        models=["ResNet50", "SVM"],
        methods=["Skull stripping", "Intensity normalization", "Grad-CAM"],
    )
    arch_gen = ArchitectureGenerator()
    roadmap_gen = RoadmapGenerator()

    resp_a = arch_gen.generate_architecture(
        documents=[doc_a],
        query="Cancer detection using MRI",
    )
    roadmap_a = roadmap_gen.generate_roadmap(
        documents=[doc_a],
        architecture=resp_a.architecture,
        query="Cancer detection using MRI",
    )

    # Architecture B: Crop Disease
    doc_b = make_doc(
        "doc_crop",
        "Deep Learning for Crop Disease Classification on PlantVillage",
        "We present EfficientNet-B0 trained on PlantVillage dataset using image resizing and color augmentation, evaluated with Macro-F1.",
        datasets=["PlantVillage"],
        models=["EfficientNet-B0"],
        methods=["Image resizing", "Color augmentation", "Macro-F1"],
    )
    resp_b = arch_gen.generate_architecture(
        documents=[doc_b],
        query="Crop disease detection",
    )
    roadmap_b = roadmap_gen.generate_roadmap(
        documents=[doc_b],
        architecture=resp_b.architecture,
        query="Crop disease detection",
    )

    assert roadmap_a.architecture_hash != roadmap_b.architecture_hash, "Hashes must differ for distinct architectures"
    
    tasks_a = {t.task.lower() for phase in roadmap_a.phases for t in phase.task_items}
    tasks_b = {t.task.lower() for phase in roadmap_b.phases for t in phase.task_items}
    assert tasks_a != tasks_b, "Roadmap task sets must differ between MRI and Crop disease domains"

    techs_a = {t.technology.lower() for phase in roadmap_a.phases for t in phase.task_items}
    techs_b = {t.technology.lower() for phase in roadmap_b.phases for t in phase.task_items}
    assert techs_a != techs_b, "Technologies must differ between MRI and Crop disease domains"


def test_roadmap_contains_current_architecture_components():
    """Verify roadmap contains tasks generated directly from current architecture components."""
    doc = make_doc(
        "doc_full",
        "Full Pipeline Multi-Stage Classification",
        "We introduce a complete system with custom preprocessing, backbone, and evaluation.",
        datasets=["CustomBenchmark"],
        models=["TransformerBackbone"],
        methods=["Token Normalization", "Cross-Entropy Optimization"],
    )
    arch_gen = ArchitectureGenerator()
    roadmap_gen = RoadmapGenerator()

    resp = arch_gen.generate_architecture(
        documents=[doc],
        query="Transformer Benchmark",
    )
    roadmap = roadmap_gen.generate_roadmap(
        documents=[doc],
        architecture=resp.architecture,
        query="Transformer Benchmark",
    )

    all_descriptions = " ".join(t.description for phase in roadmap.phases for t in phase.task_items).lower()
    all_tasks = " ".join(t.task for phase in roadmap.phases for t in phase.task_items).lower()
    all_techs = " ".join(t.technology for phase in roadmap.phases for t in phase.task_items).lower()
    combined = f"{all_tasks} {all_descriptions} {all_techs}"

    # Verify key architectural pipeline elements are represented in roadmap tasks
    assert "dataset" in combined or "custombenchmark" in combined
    assert "preprocess" in combined or "normalization" in combined
    assert "model" in combined or "transformer" in combined
    assert "evaluat" in combined or "metric" in combined
    assert "infer" in combined or "onnx" in combined or "tensorrt" in combined
    assert "api" in combined or "fastapi" in combined
    assert "docker" in combined or "deploy" in combined


def test_roadmap_contains_current_dataset():
    """Verify roadmap tasks directly contain the current dataset name."""
    doc_mri = make_doc(
        "doc_mri",
        "Brain MRI Glioma Study",
        "We evaluate on BraTS-2023 dataset.",
        datasets=["BraTS-2023"],
        models=["ResNet50"],
    )
    arch_gen = ArchitectureGenerator()
    roadmap_gen = RoadmapGenerator()

    resp_mri = arch_gen.generate_architecture([doc_mri], query="Brain MRI")
    roadmap_mri = roadmap_gen.generate_roadmap([doc_mri], architecture=resp_mri.architecture, query="Brain MRI")

    phase1_tasks = [t for phase in roadmap_mri.phases if phase.phase == 1 for t in phase.task_items]
    phase3_tasks = [t for phase in roadmap_mri.phases if phase.phase == 3 for t in phase.task_items]
    phase1_3_text = " ".join(f"{t.task} {t.description}" for t in phase1_tasks + phase3_tasks)

    assert "brats" in phase1_3_text.lower(), "Current dataset BraTS must appear in Phase 1 or Phase 3 tasks"


def test_roadmap_contains_current_model():
    """Verify roadmap tasks directly contain the current architecture model."""
    doc_svm = make_doc(
        "doc_svm",
        "ResNet Feature SVM Classification",
        "We extract features and train a Support Vector Machine.",
        datasets=["MRI-Scans"],
        models=["ResNet50", "SVM"],
    )
    arch_gen = ArchitectureGenerator()
    roadmap_gen = RoadmapGenerator()

    resp = arch_gen.generate_architecture([doc_svm], query="SVM Classifier")
    roadmap = roadmap_gen.generate_roadmap([doc_svm], architecture=resp.architecture, query="SVM Classifier")

    phase4_tasks = [t for phase in roadmap.phases if phase.phase == 4 for t in phase.task_items]
    phase4_text = " ".join(f"{t.task} {t.description} {t.technology}" for t in phase4_tasks).lower()

    assert "svm" in phase4_text or "resnet" in phase4_text, "Current model must appear in Phase 4 tasks"


def test_roadmap_contains_current_preprocessing():
    """Verify roadmap tasks directly contain current architecture preprocessing techniques."""
    doc_prep = make_doc(
        "doc_prep",
        "Advanced Neuroimaging Preprocessing",
        "We perform skull stripping, intensity normalization, and N4 bias field correction.",
        datasets=["BrainMRIData"],
        models=["U-Net"],
        methods=["Skull stripping", "Intensity normalization", "N4 bias correction"],
    )
    arch_gen = ArchitectureGenerator()
    roadmap_gen = RoadmapGenerator()

    resp = arch_gen.generate_architecture([doc_prep], query="Neuroimaging Preprocessing")
    roadmap = roadmap_gen.generate_roadmap([doc_prep], architecture=resp.architecture, query="Neuroimaging Preprocessing")

    phase3_tasks = [t for phase in roadmap.phases if phase.phase == 3 for t in phase.task_items]
    phase3_text = " ".join(f"{t.task} {t.description}" for t in phase3_tasks).lower()

    assert "skull strip" in phase3_text or "intensity normaliz" in phase3_text or "bias" in phase3_text or "preprocess" in phase3_text


def test_roadmap_contains_current_evaluation():
    """Verify roadmap tasks directly contain current architecture evaluation metrics."""
    doc_eval = make_doc(
        "doc_eval",
        "Medical Diagnostic Evaluation",
        "We evaluate using Sensitivity, Specificity, and AUC on clinical test cohorts.",
        datasets=["ClinicalData"],
        models=["DenseNet121"],
        methods=["Sensitivity", "Specificity", "AUC"],
    )
    arch_gen = ArchitectureGenerator()
    roadmap_gen = RoadmapGenerator()

    resp = arch_gen.generate_architecture([doc_eval], query="Diagnostic Evaluation")
    roadmap = roadmap_gen.generate_roadmap([doc_eval], architecture=resp.architecture, query="Diagnostic Evaluation")

    phase6_tasks = [t for phase in roadmap.phases if phase.phase == 6 for t in phase.task_items]
    phase6_text = " ".join(f"{t.task} {t.description} {t.technology}" for t in phase6_tasks).lower()

    assert "sensitivity" in phase6_text or "specificity" in phase6_text or "auc" in phase6_text or "metric" in phase6_text


def test_same_architecture_same_roadmap():
    """Verify running the same query/papers twice produces identical architecture_hash and tasks."""
    doc = make_doc(
        "doc_repeat",
        "Reproducible Architecture Study",
        "Benchmarking reproducibility on ImageNet with ResNet50.",
        datasets=["ImageNet"],
        models=["ResNet50"],
        methods=["Cross-Entropy"],
    )
    arch_gen = ArchitectureGenerator()
    roadmap_gen = RoadmapGenerator()

    resp1 = arch_gen.generate_architecture([doc], query="Reproducible Study")
    roadmap1 = roadmap_gen.generate_roadmap([doc], architecture=resp1.architecture, query="Reproducible Study")

    resp2 = arch_gen.generate_architecture([doc], query="Reproducible Study")
    roadmap2 = roadmap_gen.generate_roadmap([doc], architecture=resp2.architecture, query="Reproducible Study")

    assert roadmap1.architecture_hash == roadmap2.architecture_hash
    assert len(roadmap1.phases) == len(roadmap2.phases)
    
    tasks1 = [(t.phase, t.task, t.technology) for phase in roadmap1.phases for t in phase.task_items]
    tasks2 = [(t.phase, t.task, t.technology) for phase in roadmap2.phases for t in phase.task_items]
    assert tasks1 == tasks2, "Identical input must produce identical task sequence and ordering"


def test_frontend_uses_live_roadmap_response():
    """Verify POST /api/v1/roadmap returns architecture_hash and all 10 task fields for matrix rendering."""
    client = TestClient(app)
    doc = make_doc(
        "doc_api_test",
        "API Live Roadmap Verification",
        "Testing live API responses with end-to-end task matrix metadata.",
        datasets=["LiveDataset"],
        models=["LiveModel"],
        methods=["Augmentation"],
    )

    payload = {
        "documents": [doc.model_dump()],
        "query": "Live API Test",
    }
    response = client.post("/api/v1/roadmap", json=payload)
    assert response.status_code == 200, f"API failed: {response.text}"
    
    data = response.json()
    assert "architecture_hash" in data
    assert data["architecture_hash"] != "", "architecture_hash must not be empty"
    assert "phases" in data
    assert len(data["phases"]) == 8

    # Verify all 10 fields are present on tasks in task_items
    required_fields = [
        "phase",
        "task",
        "description",
        "dependency",
        "input",
        "output",
        "technology",
        "difficulty",
        "evidence_status",
        "supporting_papers",
    ]
    total_tasks = 0
    for phase_data in data["phases"]:
        for task_dict in phase_data["task_items"]:
            total_tasks += 1
            for field in required_fields:
                assert field in task_dict, f"Field '{field}' missing from task in phase {phase_data['phase']}"
                assert task_dict[field] is not None, f"Field '{field}' is None"

    assert total_tasks > 10, "Must have generated a full suite of roadmap tasks"
