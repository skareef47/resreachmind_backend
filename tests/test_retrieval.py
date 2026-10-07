import asyncio
from typing import List
import pytest

from app.retrieval.bm25 import BM25Retriever, tokenize_text
from app.retrieval.pipeline import HybridRetrievalPipeline
from app.retrieval.rrf import RRFCombiner
from app.retrieval.sbert import SBERTRetriever, cosine_similarity
from app.schemas.paper import ResearchDocument


@pytest.fixture
def sample_corpus() -> List[ResearchDocument]:
    return [
        ResearchDocument(
            internal_id="doc_1",
            source="openalex",
            source_id="W1",
            title="Attention Is All You Need",
            abstract="The Transformer network architecture based on self-attention mechanisms.",
            concepts=["Transformer", "Attention mechanism"],
            doi="10.48550/arxiv.1706.03762",
            authors=["Vaswani et al."],
            publication_year=2017,
        ),
        ResearchDocument(
            internal_id="doc_2",
            source="semantic_scholar",
            source_id="s2_2",
            title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            abstract="We introduce BERT, bidirectional encoder representations from transformers for NLP tasks.",
            concepts=["BERT", "NLP", "Language Models"],
            doi="10.18653/v1/n19-1423",
            authors=["Devlin et al."],
            publication_year=2019,
        ),
        ResearchDocument(
            internal_id="doc_3",
            source="semantic_scholar",
            source_id="s2_3",
            title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            abstract="RAG models combine parametric and non-parametric memory for question answering.",
            concepts=["RAG", "Dense Retrieval", "LLM"],
            doi="10.5555/rag.2020",
            authors=["Lewis et al."],
            publication_year=2020,
        ),
        ResearchDocument(
            internal_id="doc_4",
            source="openalex",
            source_id="W4",
            title="Cross-Encoder Reranking in Neural Information Retrieval",
            abstract="Using a cross-encoder model for fine-grained re-ranking of candidate document pools.",
            concepts=["cross-encoder", "re-ranking", "Information Retrieval"],
            doi="10.1000/cross.encoder",
            authors=["Nogueira et al."],
            publication_year=2021,
        ),
        ResearchDocument(
            internal_id="doc_5",
            source="openalex",
            source_id="W5",
            title="Convolutional Neural Networks for Audio Spectrogram Classification",
            abstract="Deep acoustic models using CNN architectures for environmental sound detection.",
            concepts=["Audio", "CNN"],
            doi="10.1000/audio.cnn",
            authors=["Piczak"],
            publication_year=2015,
        ),
    ]


# ============================================================================
# 1. BM25 RETRIEVAL TESTS
# ============================================================================

def test_bm25_tokenization_acronyms_and_technical_terms() -> None:
    tokens = tokenize_text("BERT and cross-encoder RAG models with DOI 10.1000/182")
    assert "bert" in tokens
    assert "rag" in tokens
    assert "cross-encoder" in tokens
    assert "cross" in tokens
    assert "encoder" in tokens
    assert "10.1000/182" in tokens


def test_bm25_exact_technical_term_retrieval(sample_corpus: List[ResearchDocument]) -> None:
    retriever = BM25Retriever()
    results = retriever.rank("cross-encoder", sample_corpus)
    assert len(results) == len(sample_corpus)
    # Document 4 has exact "cross-encoder" in title and abstract
    top_doc = results[0]
    assert top_doc.document.internal_id == "doc_4"
    assert top_doc.score > 0.0
    assert top_doc.rank == 1


def test_bm25_acronym_and_rare_terms(sample_corpus: List[ResearchDocument]) -> None:
    retriever = BM25Retriever()
    results = retriever.rank("RAG NLP", sample_corpus)
    # Document 3 is about RAG NLP
    top_doc = results[0]
    assert top_doc.document.internal_id == "doc_3"
    assert top_doc.rank == 1


def test_bm25_identifier_match(sample_corpus: List[ResearchDocument]) -> None:
    retriever = BM25Retriever()
    results = retriever.rank("10.18653/v1/n19-1423", sample_corpus)
    top_doc = results[0]
    assert top_doc.document.internal_id == "doc_2"
    assert top_doc.score > 0.0


def test_bm25_empty_query_and_empty_corpus(sample_corpus: List[ResearchDocument]) -> None:
    retriever = BM25Retriever()
    assert retriever.rank("query", []) == []

    # Empty query returns 0 score without error
    results = retriever.rank("", sample_corpus)
    assert len(results) == len(sample_corpus)
    assert results[0].score == 0.0


# ============================================================================
# 2. SBERT SEMANTIC RETRIEVAL TESTS
# ============================================================================

def test_cosine_similarity() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0
    assert cosine_similarity([], []) == 0.0


def test_sbert_semantic_retrieval(sample_corpus: List[ResearchDocument]) -> None:
    sbert = SBERTRetriever()
    results = sbert.rank("acoustic sound spectrograms", sample_corpus)
    assert len(results) == len(sample_corpus)
    # Document 5 is about acoustic sound spectrograms
    assert results[0].document.internal_id == "doc_5"
    assert results[0].score > 0.0


def test_sbert_deterministic_behavior(sample_corpus: List[ResearchDocument]) -> None:
    sbert = SBERTRetriever()
    text = "Neural network transformer architectures"
    emb1 = sbert.encode([text])[0]
    emb2 = sbert.encode([text])[0]
    assert emb1 == emb2

    rank1 = [r.document.internal_id for r in sbert.rank(text, sample_corpus)]
    rank2 = [r.document.internal_id for r in sbert.rank(text, sample_corpus)]
    assert rank1 == rank2


# ============================================================================
# 3. RRF COMBINATION TESTS
# ============================================================================

def test_rrf_scoring_formula(sample_corpus: List[ResearchDocument]) -> None:
    bm25 = BM25Retriever()
    sbert = SBERTRetriever()
    rrf = RRFCombiner(k=60, bm25_weight=1.0, sbert_weight=1.0)

    bm25_res = bm25.rank("transformer", sample_corpus)
    sbert_res = sbert.rank("transformer", sample_corpus)

    candidates = rrf.combine(bm25_res, sbert_res, candidate_limit=5)
    assert len(candidates) == len(sample_corpus)

    top_cand = candidates[0]
    assert top_cand.metadata.rrf_rank == 1
    assert top_cand.metadata.bm25_rank is not None
    assert top_cand.metadata.sbert_rank is not None

    # Expected formula check: 1/(k + r_bm25) + 1/(k + r_sbert)
    expected_score = 1.0 / (60 + top_cand.metadata.bm25_rank) + 1.0 / (60 + top_cand.metadata.sbert_rank)
    assert abs(top_cand.metadata.rrf_score - round(expected_score, 6)) < 1e-4


def test_rrf_combines_without_raw_score_scale(sample_corpus: List[ResearchDocument]) -> None:
    """Ensure RRF is rank-based and invariant to arbitrary scale of BM25 vs SBERT scores."""
    bm25 = BM25Retriever()
    sbert = SBERTRetriever()
    rrf = RRFCombiner(k=60)

    # Even if BM25 scores are in range [0, 50] and SBERT in [0, 1], RRF uses only ranks
    bm25_res = bm25.rank("RAG memory", sample_corpus)
    sbert_res = sbert.rank("RAG memory", sample_corpus)

    candidates = rrf.combine(bm25_res, sbert_res)
    assert len(candidates) > 0
    for idx, c in enumerate(candidates, start=1):
        assert c.metadata.rrf_rank == idx
        if idx > 1:
            assert candidates[idx - 2].metadata.rrf_score >= c.metadata.rrf_score


# ============================================================================
# 4. PIPELINE, BOUNDED LIMITS & DETERMINISM
# ============================================================================

def test_hybrid_pipeline_candidate_limits(sample_corpus: List[ResearchDocument]) -> None:
    async def _run() -> None:
        pipeline = HybridRetrievalPipeline(default_candidate_limit=2)
        candidates = await pipeline.retrieve(
            query="transformers",
            documents=sample_corpus,
            candidate_limit=2,
        )
        assert len(candidates) == 2
        assert candidates[0].metadata.rrf_rank == 1
        assert candidates[1].metadata.rrf_rank == 2

    asyncio.run(_run())


def test_hybrid_pipeline_empty_corpus() -> None:
    async def _run() -> None:
        pipeline = HybridRetrievalPipeline()
        candidates = await pipeline.retrieve(query="test", documents=[])
        assert candidates == []

    asyncio.run(_run())


def test_hybrid_pipeline_deterministic_behavior(sample_corpus: List[ResearchDocument]) -> None:
    async def _run() -> None:
        pipeline = HybridRetrievalPipeline()
        c1 = await pipeline.retrieve(query="deep bidirectional transformers", documents=sample_corpus)
        c2 = await pipeline.retrieve(query="deep bidirectional transformers", documents=sample_corpus)

        ids1 = [c.document.internal_id for c in c1]
        ids2 = [c.document.internal_id for c in c2]
        assert ids1 == ids2

        scores1 = [c.metadata.rrf_score for c in c1]
        scores2 = [c.metadata.rrf_score for c in c2]
        assert scores1 == scores2

    asyncio.run(_run())
