"""
Comprehensive tests for Research Gap Intelligence in ResearchMind AI.

Validates:
1. Gap analysis is grounded in actual candidate research papers
2. Supporting papers actually exist in the candidate set (zero fabricated citations)
3. Direct evidence is attached to every identified gap
4. Different paper sets produce different gap analyses
5. Terminology compliance: no claim of global novelty ("Limited evidence was found within the analyzed candidate set.")
6. Evidence-backed observations are separated from AI-generated hypotheses
7. POST /api/v1/research-gaps endpoint works
8. GET /api/v1/research-gaps endpoint works
9. Frontend contract compatibility: all expected keys exist
10. End-to-end retrieval to gap analysis pipeline
"""

import pytest
from fastapi.testclient import TestClient

from app.intelligence.gaps import ResearchGapAnalyzer
from app.intelligence.models import (
    GapType,
    PotentialResearchGap,
    ResearchGapsResponse,
)
from app.main import app
from app.schemas.paper import ResearchDocument


def make_gap_test_doc(
    doc_id: str,
    title: str,
    abstract: str = "",
    authors: list = None,
    datasets: list = None,
    models: list = None,
    methods: list = None,
    code_url: str = None,
    venue: str = "NeurIPS",
    publication_year: int = 2023,
    unknown_fields: list = None,
) -> ResearchDocument:
    return ResearchDocument(
        internal_id=f"doc_{doc_id}",
        source="openalex",
        source_id=doc_id,
        title=title,
        abstract=abstract or f"We present {title} with empirical experiments and standard benchmarks.",
        authors=authors or ["Alice Researcher", "Bob Engineer"],
        publication_year=publication_year,
        venue=venue,
        concepts=["Machine Learning", "Deep Learning"],
        datasets=datasets or [],
        models=models or [],
        methods=methods or [],
        code_url=code_url,
        license="MIT",
        doi=f"10.1000/{doc_id}",
        unknown_fields=unknown_fields or [],
    )


# -----------------------------------------------------------------------------
# 1. GROUNDING & EVIDENCE TESTS
# -----------------------------------------------------------------------------

def test_gap_analysis_uses_actual_papers():
    """Verify that gap analysis strictly analyzes provided candidate papers."""
    doc1 = make_gap_test_doc(
        "p1",
        "Attention Is All You Need",
        abstract="We propose the Transformer model based on self-attention. However, a major limitation is quadratic memory complexity with sequence length.",
        datasets=["WMT 2014"],
        models=["Transformer"],
        methods=["Self-Attention"],
    )
    doc2 = make_gap_test_doc(
        "p2",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        abstract="We introduce BERT for language representations evaluated on GLUE. Training requires expensive compute clusters and long pre-training runs.",
        datasets=["GLUE", "BooksCorpus"],
        models=["BERT"],
        methods=["Masked Language Modeling", "Fine-Tuning"],
    )

    analyzer = ResearchGapAnalyzer()
    res = analyzer.analyze_gaps([doc1, doc2], query="transformer attention")

    assert isinstance(res, ResearchGapsResponse)
    assert res.candidate_papers_analyzed == 2
    assert len(res.potential_gaps) >= 2

    # Check supporting papers exist and match doc IDs
    valid_ids = {"doc_p1", "doc_p2"}
    for gap in res.potential_gaps:
        assert len(gap.supporting_papers) >= 1
        for ref in gap.supporting_papers:
            assert ref.id in valid_ids, f"Fabricated paper ID detected: {ref.id}"
            assert ref.title in {doc1.title, doc2.title}


def test_evidence_is_attached_to_every_gap():
    """Ensure every identified potential gap contains direct concrete textual or statistical evidence."""
    doc1 = make_gap_test_doc(
        "ev1",
        "Deep Residual Learning for Image Recognition",
        abstract="Evaluated on ImageNet 2012 classification. However, a primary limitation remains significant degradation under adversarial perturbations and high compute requirements.",
        datasets=["ImageNet"],
        models=["ResNet50"],
        methods=["Residual Connections"],
    )
    doc2 = make_gap_test_doc(
        "ev2",
        "Densely Connected Convolutional Networks",
        abstract="DenseNet connects each layer to every other layer. Yet, memory consumption during training is expensive and testing is restricted to clean natural images.",
        datasets=["ImageNet", "CIFAR-10"],
        models=["DenseNet"],
        methods=["Feature Reuse"],
    )

    analyzer = ResearchGapAnalyzer()
    res = analyzer.analyze_gaps([doc1, doc2])

    for gap in res.potential_gaps:
        assert isinstance(gap, PotentialResearchGap)
        assert len(gap.evidence) >= 1
        # Evidence must quote or reference paper
        assert any("ResNet" in ev or "DenseNet" in ev or "Paper" in ev or "ImageNet" in ev or "'" in ev for ev in gap.evidence)
        assert gap.confidence > 0.0


def test_no_fabricated_citations():
    """Verify that paper IDs and citations in supporting_papers are drawn strictly from input corpus."""
    docs = [
        make_gap_test_doc(f"strict_{i}", f"Study {i} on Optimization", abstract=f"Abstract of study {i} discussing convergence.")
        for i in range(4)
    ]
    analyzer = ResearchGapAnalyzer()
    res = analyzer.analyze_gaps(docs)

    known_ids = {d.internal_id for d in docs}
    known_titles = {d.title for d in docs}

    for gap in res.potential_gaps:
        for ref in gap.supporting_papers:
            assert ref.id in known_ids
            assert ref.title in known_titles


# -----------------------------------------------------------------------------
# 2. EPISTEMOLOGICAL INTEGRITY & TERMINOLOGY
# -----------------------------------------------------------------------------

def test_terminology_and_epistemological_disclaimer():
    """
    Ensure the system never claims objective global novelty across all literature,
    and uses required qualifiers.
    """
    doc = make_gap_test_doc(
        "qual1",
        "Scaling Laws for Neural Language Models",
        abstract="We empirically study empirical scaling laws for language model performance with compute.",
    )
    analyzer = ResearchGapAnalyzer()
    res = analyzer.analyze_gaps([doc])

    # Landscape summary scope statement
    assert "Limited evidence was found within the analyzed candidate set" in res.landscape_summary.coverage_analysis_scope
    assert "Limited evidence was found within the analyzed candidate set" in res.disclaimer

    # Gap titles must start with Potential Research Gap or Underexplored Area
    for gap in res.potential_gaps:
        assert (
            gap.title.startswith("Potential Research Gap")
            or gap.title.startswith("Underexplored Area")
            or gap.title.startswith("Research Opportunity")
        )
        assert "Limited evidence was found within the analyzed candidate set" in gap.observation
        # Check that observation and hypothesis are distinct
        assert gap.observation != gap.hypothesis
        assert len(gap.possible_research_direction) > 10


def test_evidence_backed_observation_vs_hypothesis_distinction():
    """Check that observation describes candidate corpus facts, while hypothesis proposes a testable direction."""
    doc1 = make_gap_test_doc(
        "obs1",
        "Speech Recognition with Conformer",
        abstract="Evaluated on LibriSpeech clean test set. Struggles with noisy acoustic channels and high inference latency.",
        datasets=["LibriSpeech"],
        models=["Conformer"],
    )
    doc2 = make_gap_test_doc(
        "obs2",
        "Wav2Vec 2.0 Self-Supervised Speech",
        abstract="Evaluated on LibriSpeech without background noise. A major bottleneck remains real-time streaming inference latency.",
        datasets=["LibriSpeech"],
        models=["Wav2Vec2"],
    )

    analyzer = ResearchGapAnalyzer()
    res = analyzer.analyze_gaps([doc1, doc2])

    for gap in res.potential_gaps:
        # Observation reports empirical presence/absence
        assert "Within the" in gap.observation or "candidate set" in gap.observation or "Limited evidence" in gap.observation
        # Hypothesis proposes speculative potential
        assert any(w in gap.hypothesis.lower() for w in ["may", "could", "investigation", "applying", "mitigate", "achieve"])
        # Research direction specifies an experiment
        assert any(w in gap.possible_research_direction.lower() for w in ["conduct", "design", "investigate", "curate", "benchmark", "implement"])


# -----------------------------------------------------------------------------
# 3. SENSITIVITY TO INPUT CORPUS
# -----------------------------------------------------------------------------

def test_different_paper_sets_produce_different_gaps():
    """Different thematic sets (e.g. computer vision vs NLP) produce distinct gap analyses."""
    vision_docs = [
        make_gap_test_doc("v1", "ViT Vision Transformer", datasets=["ImageNet-21k"], models=["ViT"]),
        make_gap_test_doc("v2", "Swin Transformer", datasets=["ImageNet"], models=["Swin"]),
    ]
    speech_docs = [
        make_gap_test_doc("s1", "Whisper Speech Encoder", datasets=["LibriSpeech"], models=["Whisper"]),
        make_gap_test_doc("s2", "WavLM Speech Benchmark", datasets=["TIMIT"], models=["WavLM"]),
    ]

    analyzer = ResearchGapAnalyzer()
    res_vis = analyzer.analyze_gaps(vision_docs, query="vision transformers")
    res_speech = analyzer.analyze_gaps(speech_docs, query="speech recognition")

    # Datasets in landscape summary must differ
    assert "ImageNet" in str(res_vis.landscape_summary.dominant_datasets) or "ImageNet-21k" in str(res_vis.landscape_summary.dominant_datasets)
    assert "LibriSpeech" in str(res_speech.landscape_summary.dominant_datasets) or "TIMIT" in str(res_speech.landscape_summary.dominant_datasets)

    # Gap descriptions must reflect different domains
    vis_text = " ".join([g.title + " " + g.description for g in res_vis.potential_gaps])
    speech_text = " ".join([g.title + " " + g.description for g in res_speech.potential_gaps])

    assert "ViT" in vis_text or "ImageNet" in vis_text
    assert "Whisper" in speech_text or "LibriSpeech" in speech_text or "WavLM" in speech_text


# -----------------------------------------------------------------------------
# 4. API VERIFICATION
# -----------------------------------------------------------------------------

def test_api_research_gaps_post_with_documents():
    """POST /api/v1/research-gaps accepts documents and query, returning 200 with structured gaps."""
    client = TestClient(app)
    doc1 = make_gap_test_doc(
        "api_p1",
        "BERT for Natural Language Understanding",
        abstract="Pre-trained on English Wikipedia. A key limitation is quadratic attention cost and lack of cross-lingual zero-shot evaluation.",
        datasets=["GLUE"],
        models=["BERT"],
    )
    doc2 = make_gap_test_doc(
        "api_p2",
        "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        abstract="Trained on larger datasets. Struggles with high memory footprint during long-document fine-tuning.",
        datasets=["GLUE", "SQuAD"],
        models=["RoBERTa"],
    )

    payload = {
        "query": "bert nlp",
        "paper_ids": [doc1.internal_id, doc2.internal_id],
        "documents": [doc1.model_dump(), doc2.model_dump()],
    }

    resp = client.post("/api/v1/research-gaps", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["candidate_papers_analyzed"] == 2
    assert "potential_gaps" in data
    assert len(data["potential_gaps"]) >= 2
    assert "landscape_summary" in data
    assert "disclaimer" in data

    # Verify schema of the first gap
    g0 = data["potential_gaps"][0]
    assert "title" in g0
    assert "description" in g0
    assert "gap_type" in g0
    assert "confidence" in g0
    assert "evidence" in g0
    assert "supporting_papers" in g0
    assert "why_it_appears" in g0
    assert "observation" in g0
    assert "hypothesis" in g0
    assert "possible_research_direction" in g0


def test_api_research_gaps_get_endpoint():
    """GET /api/v1/research-gaps?query=... performs dynamic retrieval and returns gaps."""
    client = TestClient(app)
    resp = client.get("/api/v1/research-gaps?query=attention+transformer")
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["candidate_papers_analyzed"] > 0
    assert len(data["potential_gaps"]) >= 1
    assert "landscape_summary" in data


# -----------------------------------------------------------------------------
# 5. FRONTEND CONTRACT & END-TO-END PIPELINE
# -----------------------------------------------------------------------------

def test_frontend_data_contract_matches_ui_requirements():
    """Ensure response schema matches all elements required by frontend view."""
    doc = make_gap_test_doc("ui_contract", "Sample Paper on Neural Graph Reasoning")
    analyzer = ResearchGapAnalyzer()
    res = analyzer.analyze_gaps([doc])
    dump = res.model_dump()

    # Required top-level keys
    for key in ["query", "candidate_papers_analyzed", "potential_gaps", "landscape_summary", "disclaimer"]:
        assert key in dump

    # Required gap item keys
    for g in dump["potential_gaps"]:
        for field in [
            "id",
            "title",
            "description",
            "gap_type",
            "confidence",
            "evidence",
            "supporting_papers",
            "why_it_appears",
            "observation",
            "hypothesis",
            "possible_research_direction",
        ]:
            assert field in g, f"Missing field in potential_gap: {field}"


def test_end_to_end_search_to_gap_analysis_pipeline():
    """Simulate real search results being passed into POST /api/v1/research-gaps."""
    client = TestClient(app)

    # 1. Retrieve candidates
    search_resp = client.post("/api/v1/search", json={
        "query": "resnet deep residual",
        "limit": 3,
        "sources": ["openalex"],
        "optimize": False,
    })
    assert search_resp.status_code == 200, search_resp.text
    search_data = search_resp.json()
    docs = search_data.get("results", [])
    assert len(docs) > 0

    doc_ids = [d["internal_id"] for d in docs]

    # 2. Run gap analysis
    gap_resp = client.post("/api/v1/research-gaps", json={
        "query": "resnet deep residual",
        "paper_ids": doc_ids,
        "documents": docs,
    })
    assert gap_resp.status_code == 200, gap_resp.text
    gap_data = gap_resp.json()

    assert gap_data["candidate_papers_analyzed"] == len(docs)
    assert len(gap_data["potential_gaps"]) >= 2
    # Supporting papers must correspond to real IDs
    for gap in gap_data["potential_gaps"]:
        for ref in gap["supporting_papers"]:
            assert ref["id"] in doc_ids
