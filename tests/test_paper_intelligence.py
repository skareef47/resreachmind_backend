"""
Comprehensive tests for Paper Intelligence Engine in ResearchMind AI.

Validates:
1. Paper Intelligence endpoint works (POST /api/v1/paper-intelligence).
2. Paper A produces Paper A-specific analysis (MRI cancer detection).
3. Paper B produces Paper B-specific analysis (Crop disease detection).
4. Paper A and Paper B produce different limitations when their evidence differs.
5. Mitigations correspond to the selected paper's limitations and distinguish AUTHOR vs RESEARCHMIND proposed.
6. Research gaps correspond to the selected paper and avoid 'no one has researched this'.
7. Research opportunities correspond to the selected paper.
8. Missing evidence becomes UNKNOWN ('Insufficient evidence in available paper content').
9. Same paper produces deterministic output.
10. Paper switching test: analyzing Paper B produces fresh state distinct from Paper A.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.paper import ResearchDocument
from app.intelligence.paper_intelligence import PaperIntelligenceAnalyzer
from app.schemas.paper_intelligence import (
    LimitationType,
    EvidenceStatus,
    ProposalType,
    PaperIntelligenceResponse,
)


def make_test_paper(
    doc_id: str,
    title: str,
    abstract: str,
    datasets: list = None,
    models: list = None,
    methods: list = None,
    code_url: str = None,
    doi: str = None,
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
        concepts=["Machine Learning"],
        datasets=datasets or [],
        models=models or [],
        methods=methods or [],
        code_url=code_url,
        license="MIT",
        doi=doi or f"10.1000/{doc_id}",
    )


def test_paper_intelligence_endpoint_works():
    """Verify POST /api/v1/paper-intelligence returns complete structured response."""
    client = TestClient(app)
    paper = make_test_paper(
        "p_001",
        "Multimodal Brain Tumor Segmentation with ResNet50",
        "We evaluate on the BraTS MRI dataset using skull stripping and intensity normalization with ResNet50 backbone. Our method achieved 92.4% accuracy.",
        datasets=["BraTS MRI"],
        models=["ResNet50"],
        methods=["Skull stripping", "Intensity normalization"],
        code_url="https://github.com/example/brats-resnet",
    )

    payload = {
        "paper_id": paper.internal_id,
        "paper": paper.model_dump(),
    }
    response = client.post("/api/v1/paper-intelligence", json=payload)
    assert response.status_code == 200, f"Endpoint failed: {response.text}"

    data = response.json()
    assert data["paper_id"] == paper.internal_id
    assert data["title"] == paper.title
    assert len(data["summary"]) > 10
    assert len(data["research_problem"]) > 5
    assert len(data["dataset"]) >= 1
    assert len(data["methodology"]) >= 1
    assert len(data["model"]) >= 1
    assert len(data["results"]) >= 1
    assert len(data["limitations"]) >= 1
    assert len(data["mitigations"]) >= 1
    assert len(data["research_gaps"]) >= 1
    assert len(data["opportunities"]) >= 1
    assert len(data["evidence"]) >= 1


def test_paper_a_produces_paper_a_specific_analysis():
    """Verify Paper A (MRI Cancer) produces analysis specific to MRI, BraTS, and Skull stripping."""
    analyzer = PaperIntelligenceAnalyzer()
    paper_a = make_test_paper(
        "mri_01",
        "Deep Brain MRI Glioma Detection and Skull Stripping",
        "We address the challenge of glioma detection in volumetric MRI scans. We apply skull stripping and intensity normalization on BraTS dataset with ResNet50. Results demonstrate superior 94.2% sensitivity.",
        datasets=["BraTS"],
        models=["ResNet50"],
        methods=["Skull stripping", "Intensity normalization"],
    )

    intel_a = analyzer.analyze_paper(paper_a)

    all_text = (
        f"{intel_a.summary} {intel_a.research_problem} "
        + " ".join(d.name for d in intel_a.dataset)
        + " ".join(m.name for m in intel_a.methodology)
        + " ".join(mo.name for mo in intel_a.model)
    ).lower()

    assert "mri" in all_text or "glioma" in all_text or "brats" in all_text
    assert "brats" in [d.name.lower() for d in intel_a.dataset]
    assert any("skull stripping" in m.name.lower() for m in intel_a.methodology)
    assert any("resnet" in mo.name.lower() for mo in intel_a.model)


def test_paper_b_produces_paper_b_specific_analysis():
    """Verify Paper B (Crop Disease) produces analysis specific to crops, PlantVillage, and EfficientNet."""
    analyzer = PaperIntelligenceAnalyzer()
    paper_b = make_test_paper(
        "crop_01",
        "Agricultural Crop Disease Classification using EfficientNet",
        "This paper addresses the challenge of identifying foliar plant pathology in RGB imagery. We evaluate on PlantVillage dataset using geometric augmentation and EfficientNet-B0 backbone. Achieved 98.1% accuracy.",
        datasets=["PlantVillage"],
        models=["EfficientNet-B0"],
        methods=["Geometric Augmentation"],
        code_url="https://github.com/crop/efficientnet",
    )

    intel_b = analyzer.analyze_paper(paper_b)

    all_text = (
        f"{intel_b.summary} {intel_b.research_problem} "
        + " ".join(d.name for d in intel_b.dataset)
        + " ".join(m.name for m in intel_b.methodology)
        + " ".join(mo.name for mo in intel_b.model)
    ).lower()

    assert "crop" in all_text or "plant" in all_text or "plantvillage" in all_text
    assert "plantvillage" in [d.name.lower() for d in intel_b.dataset]
    assert any("efficientnet" in mo.name.lower() for mo in intel_b.model)


def test_paper_a_and_paper_b_produce_different_limitations():
    """Verify different papers produce distinct limitations based on their actual evidence."""
    analyzer = PaperIntelligenceAnalyzer()
    paper_a = make_test_paper(
        "mri_01",
        "Brain MRI Glioma Detection",
        "We evaluate on BraTS. The method is limited by high computational cost on 3D volumetric convolutions.",
        datasets=["BraTS"],
        models=["ResNet50"],
        methods=["Skull stripping"],
        # No code_url
    )
    paper_b = make_test_paper(
        "crop_01",
        "Crop Disease Classification",
        "We evaluate on PlantVillage. The model struggles when lighting conditions vary drastically in outdoor field captures.",
        datasets=["PlantVillage"],
        models=["EfficientNet-B0"],
        methods=["Geometric Augmentation"],
        code_url="https://github.com/crop/efficientnet",
    )

    intel_a = analyzer.analyze_paper(paper_a)
    intel_b = analyzer.analyze_paper(paper_b)

    lims_a = [l.text.lower() for l in intel_a.limitations]
    lims_b = [l.text.lower() for l in intel_b.limitations]

    assert lims_a != lims_b, "Limitations must differ between different papers"
    # Paper A has no code repository
    assert any("code" in l or "implementation" in l for l in lims_a)
    # Paper B has explicit lighting limitation
    assert any("lighting" in l or "field" in l or "struggles" in l for l in lims_b)


def test_mitigations_correspond_to_limitations_and_attribution():
    """Verify mitigations address each limitation and distinguish AUTHOR vs RESEARCHMIND proposed."""
    analyzer = PaperIntelligenceAnalyzer()
    paper = make_test_paper(
        "p_lim",
        "Deep Neural Model Evaluation",
        "We evaluate on SingleBenchmark. Future work will investigate cross-domain generalization.",
        datasets=["SingleBenchmark"],
        models=["ResNet50"],
    )

    intel = analyzer.analyze_paper(paper)
    assert len(intel.mitigations) == len(intel.limitations)

    for mit, lim in zip(intel.mitigations, intel.limitations):
        assert mit.limitation_id == lim.limitation_id
        assert mit.proposal_type in [ProposalType.AUTHOR_PROPOSED, ProposalType.RESEARCHMIND_PROPOSED]
        if mit.proposal_type == ProposalType.RESEARCHMIND_PROPOSED:
            assert "ResearchMind proposes:" in mit.recommendation
            assert "Author proposes:" not in mit.recommendation
        elif mit.proposal_type == ProposalType.AUTHOR_PROPOSED:
            assert "Author proposes:" in mit.recommendation


def test_research_gaps_correspond_to_paper_and_avoid_absolutes():
    """Verify research gaps use evidence-grounded phrasing and do not state 'No one has researched this'."""
    analyzer = PaperIntelligenceAnalyzer()
    paper = make_test_paper(
        "p_gaps",
        "Clinical Anomaly Detection",
        "Evaluated exclusively on HospitalA-Cohort.",
        datasets=["HospitalA-Cohort"],
        models=["DenseNet"],
    )

    intel = analyzer.analyze_paper(paper)
    assert len(intel.research_gaps) >= 1

    for gap in intel.research_gaps:
        desc = gap.description
        assert "no one has researched" not in desc.lower()
        assert "nobody has" not in desc.lower()
        # Should contain evidence-grounded language
        assert any(phrase in desc.lower() for phrase in [
            "does not evaluate",
            "limited evidence is provided",
            "leaves an opportunity",
            "insufficient evidence",
        ])


def test_research_opportunities_correspond_to_paper():
    """Verify research opportunities provide actionable steps and impact."""
    analyzer = PaperIntelligenceAnalyzer()
    paper = make_test_paper(
        "p_opps",
        "Edge Neural Inference",
        "Evaluated on synthetic data.",
        datasets=["SyntheticBench"],
        models=["MobileNet"],
    )

    intel = analyzer.analyze_paper(paper)
    assert len(intel.opportunities) >= 1

    for opp in intel.opportunities:
        assert len(opp.title) > 3
        assert len(opp.description) > 10
        assert len(opp.actionable_steps) >= 1
        assert len(opp.expected_impact) > 5


def test_missing_evidence_becomes_unknown():
    """Verify papers with minimal/missing content report UNKNOWN rather than fabricating data."""
    analyzer = PaperIntelligenceAnalyzer()
    empty_paper = make_test_paper(
        "p_empty",
        "",  # No title
        "",  # No abstract
        datasets=[],
        models=[],
        methods=[],
        code_url=None,
    )

    intel = analyzer.analyze_paper(empty_paper)

    # Summary and problem should report insufficient evidence
    assert "Insufficient evidence" in intel.summary or "insufficient" in intel.summary.lower()
    assert "Insufficient evidence" in intel.research_problem or "insufficient" in intel.research_problem.lower()

    # Datasets, methodology, models should contain UNKNOWN status
    assert any(d.status == EvidenceStatus.UNKNOWN for d in intel.dataset)
    assert any(m.status == EvidenceStatus.UNKNOWN for m in intel.methodology)
    assert any(mo.status == EvidenceStatus.UNKNOWN for mo in intel.model)
    assert any(r.status == EvidenceStatus.UNKNOWN for r in intel.results)

    # Limitations should contain UNKNOWN
    assert any(l.type == LimitationType.UNKNOWN for l in intel.limitations)


def test_same_paper_produces_deterministic_output():
    """Verify repeated analysis of the same paper yields identical results."""
    analyzer = PaperIntelligenceAnalyzer()
    paper = make_test_paper(
        "p_repeat",
        "Deterministic Analysis Paper",
        "We evaluate on ImageNet with ResNet50 using SGD optimizer.",
        datasets=["ImageNet"],
        models=["ResNet50"],
        methods=["SGD Optimizer"],
    )

    intel1 = analyzer.analyze_paper(paper)
    intel2 = analyzer.analyze_paper(paper)

    assert intel1.summary == intel2.summary
    assert intel1.research_problem == intel2.research_problem
    assert [l.text for l in intel1.limitations] == [l.text for l in intel2.limitations]
    assert [m.recommendation for m in intel1.mitigations] == [m.recommendation for m in intel2.mitigations]
    assert [g.description for g in intel1.research_gaps] == [g.description for g in intel2.research_gaps]


def test_paper_switching_clears_and_replaces_state():
    """Verify switching from Paper A to Paper B completely isolates state."""
    client = TestClient(app)
    paper_a = make_test_paper("switch_a", "Brain MRI Paper", "We study brain MRI tumors.", datasets=["BraTS"])
    paper_b = make_test_paper("switch_b", "Crop Disease Paper", "We study plant diseases.", datasets=["PlantVillage"])

    resp_a = client.post("/api/v1/paper-intelligence", json={"paper_id": paper_a.internal_id, "paper": paper_a.model_dump()})
    assert resp_a.status_code == 200
    data_a = resp_a.json()

    resp_b = client.post("/api/v1/paper-intelligence", json={"paper_id": paper_b.internal_id, "paper": paper_b.model_dump()})
    assert resp_b.status_code == 200
    data_b = resp_b.json()

    assert data_a["paper_id"] != data_b["paper_id"]
    assert data_a["title"] != data_b["title"]
    assert data_a["summary"] != data_b["summary"]
