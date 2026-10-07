import asyncio
import pytest
from typing import List

from app.config import get_settings
from app.retrieval.features import (
    RelevanceFeatures,
    compute_recency_score,
    extract_candidate_features,
    min_max_scale,
    sigmoid,
)
from app.retrieval.pipeline import HybridRetrievalPipeline
from app.retrieval.reranker import (
    CrossEncoderReranker,
    DeterministicCrossEncoderEngine,
    FailingCrossEncoderEngine,
)
from app.schemas.paper import ResearchDocument
from app.schemas.retrieval import RetrievalMetadata, RetrievedCandidate


def _make_candidate(
    doc_id: str,
    title: str,
    abstract: str = "",
    rrf_rank: int = 1,
    rrf_score: float = 0.03,
    bm25_score: float = 2.0,
    sbert_score: float = 0.8,
    citation_count: int = 10,
    publication_year: int = 2023,
    code_url: str = "",
    open_access: bool = True,
) -> RetrievedCandidate:
    doc = ResearchDocument(
        internal_id=f"doc_{doc_id}",
        source="openalex",
        source_id=doc_id,
        title=title,
        abstract=abstract,
        authors=["Author One"],
        publication_year=publication_year,
        citation_count=citation_count,
        code_url=code_url if code_url else None,
        open_access=open_access,
    )
    meta = RetrievalMetadata(
        bm25_rank=rrf_rank,
        bm25_score=bm25_score,
        sbert_rank=rrf_rank,
        sbert_score=sbert_score,
        rrf_score=rrf_score,
        rrf_rank=rrf_rank,
    )
    return RetrievedCandidate(document=doc, metadata=meta)


def test_reranker_basic_ordering() -> None:
    """Verify Cross-Encoder reranks candidate pool based on fine-grained query-document relevance."""
    async def _run() -> None:
        reranker = CrossEncoderReranker(engine=DeterministicCrossEncoderEngine())

        candidates = [
            _make_candidate("1", "Photosynthesis in Marine Algae", "Study on phytoplankton in Pacific Ocean", rrf_rank=1, rrf_score=0.04),
            _make_candidate("2", "Dense Passage Retrieval for Open-Domain QA", "DPR uses dual-encoders for passage retrieval", rrf_rank=2, rrf_score=0.03),
            _make_candidate("3", "Survey of Neural Information Retrieval", "Overview of BM25, DPR, and cross-encoders", rrf_rank=3, rrf_score=0.02),
        ]

        reranked = await reranker.rerank(
            query="Dense Passage Retrieval QA",
            candidates=candidates,
        )

        assert len(reranked) == 3
        # Candidate 2 is the most relevant exact match and should be ranked 1st
        assert reranked[0].document.source_id == "2"
        assert reranked[0].rerank_metadata is not None
        assert reranked[0].rerank_metadata.post_rerank_rank == 1
        assert reranked[0].rerank_metadata.pre_rerank_rank == 2

        # Irrelevant candidate should be demoted to last place
        assert reranked[2].document.source_id == "1"
        assert reranked[2].rerank_metadata.post_rerank_rank == 3

    asyncio.run(_run())


def test_reranker_pre_post_rank_and_metadata() -> None:
    """Verify pre-rerank rank, post-rerank rank, scores, and model metadata are correctly populated."""
    async def _run() -> None:
        reranker = CrossEncoderReranker(
            model_name="test-cross-encoder",
            model_version="1.2.0",
            device="cpu",
            batch_size=8,
            engine=DeterministicCrossEncoderEngine(),
        )

        candidates = [
            _make_candidate("low_rrf", "Transformers for Machine Translation", "Deep attention models", rrf_rank=5, rrf_score=0.01),
            _make_candidate("high_rrf", "Graph Neural Networks for Chemistry", "Molecular graphs", rrf_rank=1, rrf_score=0.05),
        ]

        reranked = await reranker.rerank(
            query="Transformers Machine Translation",
            candidates=candidates,
        )

        assert len(reranked) == 2
        top = reranked[0]
        assert top.document.source_id == "low_rrf"
        assert top.rerank_metadata.pre_rerank_rank == 5
        assert top.rerank_metadata.post_rerank_rank == 1
        assert 0.0 <= top.rerank_metadata.reranker_score <= 1.0
        assert top.rerank_metadata.raw_score is not None

        meta = top.rerank_metadata.model_metadata
        assert meta is not None
        assert meta.model_name == "test-cross-encoder"
        assert meta.model_version == "1.2.0"
        assert meta.device == "cpu"
        assert meta.batch_size == 8
        assert meta.latency_ms >= 0.0
        assert meta.is_fallback is False

    asyncio.run(_run())


def test_reranker_batch_inference() -> None:
    """Verify batch inference produces consistent results across different batch sizes."""
    async def _run() -> None:
        engine = DeterministicCrossEncoderEngine()
        candidates = [
            _make_candidate(str(i), f"Paper {i} on Deep Learning", f"Abstract content {i}", rrf_rank=i)
            for i in range(1, 9)
        ]

        reranker_b1 = CrossEncoderReranker(batch_size=1, engine=engine)
        reranker_b4 = CrossEncoderReranker(batch_size=4, engine=engine)
        reranker_b16 = CrossEncoderReranker(batch_size=16, engine=engine)

        res_b1 = await reranker_b1.rerank("Deep Learning", candidates)
        res_b4 = await reranker_b4.rerank("Deep Learning", candidates)
        res_b16 = await reranker_b16.rerank("Deep Learning", candidates)

        assert len(res_b1) == len(res_b4) == len(res_b16) == 8
        for r1, r4, r16 in zip(res_b1, res_b4, res_b16):
            assert r1.document.source_id == r4.document.source_id == r16.document.source_id
            assert r1.rerank_metadata.reranker_score == r4.rerank_metadata.reranker_score == r16.rerank_metadata.reranker_score
            assert r1.rerank_metadata.post_rerank_rank == r4.rerank_metadata.post_rerank_rank == r16.rerank_metadata.post_rerank_rank

    asyncio.run(_run())


def test_reranker_candidate_limit() -> None:
    """Verify candidate limit bounds the returned pool to candidate_limit."""
    async def _run() -> None:
        candidates = [
            _make_candidate(str(i), f"Paper {i} on Natural Language Processing", rrf_rank=i)
            for i in range(1, 15)
        ]

        reranker = CrossEncoderReranker(candidate_limit=5, engine=DeterministicCrossEncoderEngine())
        reranked = await reranker.rerank("Natural Language Processing", candidates)

        assert len(reranked) == 5
        ranks = [c.rerank_metadata.post_rerank_rank for c in reranked]
        assert ranks == [1, 2, 3, 4, 5]

        # Explicit override candidate_limit = 2
        reranked_custom = await reranker.rerank("Natural Language", candidates, candidate_limit=2)
        assert len(reranked_custom) == 2

    asyncio.run(_run())


def test_reranker_empty_input() -> None:
    """Verify empty input candidates returns empty list immediately."""
    async def _run() -> None:
        reranker = CrossEncoderReranker(engine=DeterministicCrossEncoderEngine())
        res = await reranker.rerank("any query", [])
        assert res == []

    asyncio.run(_run())


def test_reranker_graceful_model_failure() -> None:
    """Verify that when the Cross-Encoder model fails, it falls back to pre-rerank order without crashing."""
    async def _run() -> None:
        failing_engine = FailingCrossEncoderEngine("CUDA out of memory error")
        reranker = CrossEncoderReranker(engine=failing_engine)

        candidates = [
            _make_candidate("doc_1", "First Paper", rrf_rank=1, rrf_score=0.05),
            _make_candidate("doc_2", "Second Paper", rrf_rank=2, rrf_score=0.04),
            _make_candidate("doc_3", "Third Paper", rrf_rank=3, rrf_score=0.03),
        ]

        # Must NOT raise exception!
        res = await reranker.rerank("machine learning", candidates)

        assert len(res) == 3
        # Order preserved as pre-rerank
        assert [c.document.source_id for c in res] == ["doc_1", "doc_2", "doc_3"]
        for idx, c in enumerate(res, start=1):
            assert c.rerank_metadata is not None
            assert c.rerank_metadata.post_rerank_rank == idx
            assert c.rerank_metadata.pre_rerank_rank == idx
            assert c.rerank_metadata.model_metadata.is_fallback is True
            assert c.rerank_metadata.reranker_score == 0.0

    asyncio.run(_run())


def test_score_normalization_sigmoid() -> None:
    """Verify sigmoid normalization properly bounds arbitrary model logits into [0.0, 1.0]."""
    assert sigmoid(0.0) == 0.5
    assert 0.99 < sigmoid(10.0) <= 1.0
    assert 0.0 <= sigmoid(-10.0) < 0.01

    # Stability with extreme logits
    assert sigmoid(100.0) <= 1.0
    assert sigmoid(-100.0) >= 0.0


def test_min_max_scale() -> None:
    """Verify min-max scaling normalizes list of floats."""
    scaled = min_max_scale([2.0, 4.0, 6.0])
    assert scaled == [0.0, 0.5, 1.0]

    # Identical values return min_val
    scaled_const = min_max_scale([3.0, 3.0])
    assert scaled_const == [0.0, 0.0]

    assert min_max_scale([]) == []


def test_compute_recency_score() -> None:
    """Verify publication recency computation."""
    score_current = compute_recency_score(2026, current_year=2026, half_life_years=5.0)
    assert score_current == 1.0

    score_5yr = compute_recency_score(2021, current_year=2026, half_life_years=5.0)
    assert abs(score_5yr - 0.5) < 0.01

    score_unknown = compute_recency_score(None)
    assert score_unknown == 0.5


def test_features_extraction_and_intermediate_nature() -> None:
    """
    Verify RelevanceFeatures accurately extracts intermediate features
    and treats Cross-Encoder score as an intermediate feature, not final utility.
    """
    cand = _make_candidate(
        doc_id="feat_test",
        title="Scalable Vision Models",
        citation_count=100,
        publication_year=2024,
        code_url="https://github.com/test/vision",
        open_access=True,
    )
    features = extract_candidate_features(cand, current_year=2026)

    assert "bm25_score" in features
    assert "sbert_score" in features
    assert "rrf_score" in features
    assert "cross_encoder_score" in features
    assert "citation_count" in features
    assert features["citation_count"] == 100.0
    assert "log_citation_count" in features
    assert features["log_citation_count"] > 0
    assert features["has_code"] == 1.0
    assert features["has_open_access"] == 1.0


def test_hybrid_pipeline_with_cross_encoder() -> None:
    """Verify HybridRetrievalPipeline integrates CrossEncoderReranker end-to-end."""
    async def _run() -> None:
        reranker = CrossEncoderReranker(engine=DeterministicCrossEncoderEngine())
        pipeline = HybridRetrievalPipeline(reranker=reranker)

        docs = [
            ResearchDocument(
                internal_id="doc_a",
                source="openalex",
                source_id="A",
                title="Introduction to Linear Algebra",
                abstract="Vector spaces and matrices",
            ),
            ResearchDocument(
                internal_id="doc_b",
                source="openalex",
                source_id="B",
                title="Transformer Attention Mechanisms in Deep Learning",
                abstract="Self-attention and multi-head attention mechanisms",
            ),
        ]

        # End-to-end retrieve with rerank=True
        candidates = await pipeline.retrieve("Transformer Attention Mechanisms", docs, rerank=True)

        assert len(candidates) == 2
        top = candidates[0]
        assert top.document.source_id == "B"
        assert top.rerank_metadata is not None
        assert top.rerank_metadata.post_rerank_rank == 1
        assert top.features is not None
        assert "cross_encoder_score" in top.features

        # Retrieve with rerank=False returns pre-reranked candidates
        pre_candidates = await pipeline.retrieve("Transformer Attention Mechanisms", docs, rerank=False)
        assert len(pre_candidates) == 2
        assert pre_candidates[0].rerank_metadata is None

    asyncio.run(_run())
