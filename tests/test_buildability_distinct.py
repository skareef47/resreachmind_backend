import pytest
from app.intelligence.buildability import BuildabilityEvaluator, STATUS_VERIFIED, STATUS_UNKNOWN, STATUS_ABSENT
from app.schemas.paper import ResearchDocument
from app.retrieval.features import extract_candidate_features
from app.schemas.retrieval import RetrievalMetadata, RetrievedCandidate
from app.optimization.utility import FormalUtilityModel, UtilityWeights, compute_buildability_proxy


def test_buildability_distinct_per_paper():
    """Verify distinct paper-specific buildability scores and feature breakdowns."""
    evaluator = BuildabilityEvaluator()

    # Paper 1: High reproducibility (code + datasets + models + permissive license)
    p1 = ResearchDocument(
        internal_id="doc_test_1",
        source="semantic_scholar",
        source_id="s1",
        title="Open-Source Deep Learning Framework for Mammography",
        abstract="We release an open-source deep learning framework with full code and benchmark models evaluated on the CBIS-DDSM dataset.",
        code_url="https://github.com/org/mammography-dl",
        datasets=["CBIS-DDSM", "INbreast"],
        models=["ResNet-50", "U-Net"],
        methods=["Transfer Learning", "Segmentation"],
        license="Apache-2.0",
        open_access=True,
        doi="10.1000/182",
        url="https://doi.org/10.1000/182",
    )

    # Paper 2: Unknown code and dataset (neutral non-punitive baseline, UNKNOWN != ABSENT)
    p2 = ResearchDocument(
        internal_id="doc_test_2",
        source="openalex",
        source_id="s2",
        title="Theoretical Exploration of Neural Screening",
        abstract="A mathematical and theoretical exploration of screening dynamics.",
        code_url=None,
        datasets=[],
        models=[],
        methods=[],
        license=None,
        open_access=False,
        doi=None,
        url=None,
    )

    # Paper 3: Moderate (open access, standard CNN, no public code repo)
    p3 = ResearchDocument(
        internal_id="doc_test_3",
        source="semantic_scholar",
        source_id="s3",
        title="Convolutional Neural Networks for Breast Cancer Detection",
        abstract="We investigate convolutional neural networks for breast cancer detection using clinical cohorts. Detailed experimental methodology is presented.",
        code_url=None,
        datasets=[],
        models=["CNN"],
        methods=["Classification"],
        license=None,
        open_access=True,
        doi="10.1016/j.patrec.2021.01.001",
        url="https://doi.org/10.1016/j.patrec.2021.01.001",
    )

    # Paper 4: Explicitly absent / closed proprietary artifacts
    p4 = ResearchDocument(
        internal_id="doc_test_4",
        source="openalex",
        source_id="s4",
        title="Proprietary Enterprise Screening Model",
        abstract="All code is proprietary and closed source. Training data cannot be shared due to patient privacy.",
        code_url=None,
        datasets=[],
        models=[],
        methods=[],
        license=None,
        open_access=False,
        doi=None,
        url=None,
    )

    rep1 = evaluator.evaluate_document(p1)
    rep2 = evaluator.evaluate_document(p2)
    rep3 = evaluator.evaluate_document(p3)
    rep4 = evaluator.evaluate_document(p4)

    # Assert distinct buildability scores
    assert rep1.overall_score != rep2.overall_score
    assert rep2.overall_score != rep3.overall_score
    assert rep1.overall_score > rep3.overall_score > rep2.overall_score >= rep4.overall_score

    # Assert UNKNOWN != ABSENT distinction:
    # p1 code is VERIFIED (1.0)
    assert rep1.evidence_breakdown["code"]["status"] == STATUS_VERIFIED
    assert rep1.feature_breakdown["code"] == 1.0

    # p2 code is UNKNOWN (0.40), NOT 0.0 (ABSENT)
    assert rep2.evidence_breakdown["code"]["status"] == STATUS_UNKNOWN
    assert rep2.evidence_breakdown["code"]["score"] == 0.40
    assert "Not verified from available sources" in rep2.evidence_breakdown["code"]["evidence"]

    # p4 code is explicitly ABSENT (0.05)
    assert rep4.evidence_breakdown["code"]["status"] == STATUS_ABSENT
    assert rep4.evidence_breakdown["code"]["score"] < 0.20

    # Assert dataset breakdown
    assert rep1.evidence_breakdown["dataset"]["status"] == STATUS_VERIFIED
    assert rep1.feature_breakdown["dataset"] >= 0.75
    assert rep2.evidence_breakdown["dataset"]["status"] == STATUS_UNKNOWN
    assert rep4.evidence_breakdown["dataset"]["status"] == STATUS_ABSENT

    # Assert evidence sources extracted
    assert "https://github.com/org/mammography-dl" in rep1.evidence_sources
    assert "https://doi.org/10.1000/182" in rep1.evidence_sources
    assert len(rep1.evidence_sources) >= 2

    # Assert unknown fields list
    assert "code" in rep2.unknown_fields
    assert "code" not in rep1.unknown_fields

    # Test utility model reflects distinct buildability
    cand1 = RetrievedCandidate(document=p1, metadata=RetrievalMetadata(rrf_score=0.03, rrf_rank=1))
    cand2 = RetrievedCandidate(document=p2, metadata=RetrievalMetadata(rrf_score=0.03, rrf_rank=2))

    bld1 = compute_buildability_proxy(cand1)
    bld2 = compute_buildability_proxy(cand2)
    assert bld1 != bld2
    assert bld1 == rep1.overall_score
    assert bld2 == rep2.overall_score


def test_buildability_affects_utility_score():
    """Verify that utility_score changes in direct proportion to buildability."""
    p_high_bld = ResearchDocument(
        internal_id="doc_hb",
        source="s1",
        source_id="hb",
        title="High Buildability Paper",
        abstract="Code and benchmark datasets available.",
        code_url="https://github.com/test/code",
        datasets=["ImageNet"],
        models=["ResNet"],
        open_access=True,
        license="MIT",
    )
    p_low_bld = ResearchDocument(
        internal_id="doc_lb",
        source="s2",
        source_id="lb",
        title="Low Buildability Paper",
        abstract="Theoretical analysis with private closed data.",
        code_url=None,
        datasets=[],
        models=[],
        open_access=False,
    )

    cand_hb = RetrievedCandidate(document=p_high_bld, metadata=RetrievalMetadata(rrf_score=0.03, rrf_rank=1))
    cand_lb = RetrievedCandidate(document=p_low_bld, metadata=RetrievalMetadata(rrf_score=0.03, rrf_rank=2))

    model = FormalUtilityModel()
    weights = UtilityWeights(w1_relevance=0.2, w4_buildability=0.6, w5_source_quality=0.1, w6_recency=0.1)
    ranked, _ = model.evaluate([cand_hb, cand_lb], weights=weights, persist=False)

    u_hb = next(c.features["utility_score"] for c in ranked if c.document.internal_id == "doc_hb")
    u_lb = next(c.features["utility_score"] for c in ranked if c.document.internal_id == "doc_lb")

    assert u_hb > u_lb
    assert u_hb != u_lb
