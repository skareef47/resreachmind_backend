import asyncio
import time
from typing import Any, Dict
import pytest

from app.ingestion.cache import InMemoryCache, generate_cache_key
from app.ingestion.deduplicate import (
    deduplicate_documents,
    is_duplicate,
    merge_documents,
)
from app.ingestion.normalize import (
    clean_doi,
    clean_title,
    clean_year,
    extract_code_url,
    normalize_raw_record,
    normalize_source_paper,
)
from app.schemas.paper import ProvenanceRecord, ResearchDocument
from app.schemas.source import SourcePaper


# ============================================================================
# 1. NORMALIZATION TESTS & MALFORMED RECORDS
# ============================================================================

def test_clean_title() -> None:
    assert clean_title("  Attention   Is All You Need  \n") == "Attention Is All You Need"
    assert clean_title("Deep &amp; Reinforcement Learning") == "Deep & Reinforcement Learning"
    assert clean_title("") is None
    assert clean_title(None) is None


def test_clean_doi() -> None:
    assert clean_doi("https://doi.org/10.1000/182") == "10.1000/182"
    assert clean_doi("http://dx.doi.org/10.1000/182") == "10.1000/182"
    assert clean_doi("doi:10.1000/182") == "10.1000/182"
    assert clean_doi(" 10.1000/182  ") == "10.1000/182"
    assert clean_doi("invalid_doi") is None
    assert clean_doi(None) is None


def test_clean_year() -> None:
    assert clean_year(2023, None) == 2023
    assert clean_year("2021", None) == 2021
    assert clean_year(None, "2020-05-12") == 2020
    assert clean_year("invalid", "1999-11") == 1999
    assert clean_year(None, "undated") is None


def test_extract_code_url() -> None:
    raw = {"openAccessPdf": {"url": "https://github.com/openai/gpt-3"}}
    assert extract_code_url(raw, None) == "https://github.com/openai/gpt-3"

    raw2 = {"repository": "https://gitlab.com/org/repo"}
    assert extract_code_url(raw2, None) == "https://gitlab.com/org/repo"

    assert extract_code_url({}, "https://arxiv.org/abs/1234") is None


def test_normalize_semantic_scholar_paper() -> None:
    paper = SourcePaper(
        source="semantic_scholar",
        source_id="s2_12345",
        title="  Transformers in Vision: A Survey ",
        abstract="Comprehensive review of vision models.",
        authors=["Jane Doe", "John Smith"],
        year=2022,
        publication_date="2022-04-10",
        doi="https://doi.org/10.1016/j.vis.2022.01",
        url="https://www.semanticscholar.org/paper/s2_12345",
        citation_count=180,
        venue="Computer Vision Review",
        raw_metadata={
            "fieldsOfStudy": ["Computer Science", "Artificial Intelligence"],
            "isOpenAccess": True,
            "openAccessPdf": {"url": "https://github.com/author/vision-survey"},
        },
    )

    doc = normalize_source_paper(paper)
    assert doc is not None
    assert doc.internal_id.startswith("doc_")
    assert doc.source == "semantic_scholar"
    assert doc.source_id == "s2_12345"
    assert doc.title == "Transformers in Vision: A Survey"
    assert doc.doi == "10.1016/j.vis.2022.01"
    assert doc.publication_year == 2022
    assert doc.open_access is True
    assert doc.code_url == "https://github.com/author/vision-survey"
    assert "Computer Science" in doc.fields_of_study
    assert len(doc.provenance) == 1
    assert doc.provenance[0].source == "semantic_scholar"


def test_normalize_openalex_paper() -> None:
    paper = SourcePaper(
        source="openalex",
        source_id="W99999",
        title="Foundation Models in Healthcare",
        abstract="Clinical transformers for EHR records.",
        authors=["Alice Johnson"],
        year=2023,
        publication_date="2023-08-01",
        doi="10.1038/s41591-023-01",
        citation_count=95,
        venue="Nature Medicine",
        raw_metadata={
            "concepts": [{"display_name": "Medicine"}, {"display_name": "Deep learning"}],
            "open_access": {"is_oa": True},
            "primary_location": {"license": "cc-by"},
        },
    )

    doc = normalize_source_paper(paper)
    assert doc is not None
    assert doc.source == "openalex"
    assert doc.source_id == "W99999"
    assert doc.title == "Foundation Models in Healthcare"
    assert doc.doi == "10.1038/s41591-023-01"
    assert "Medicine" in doc.concepts
    assert doc.open_access is True
    assert doc.license == "cc-by"


def test_normalize_malformed_records() -> None:
    # 1. Missing title -> None
    assert normalize_raw_record({"title": "", "id": "1"}) is None
    assert normalize_raw_record({"title": "   ", "id": "1"}) is None
    assert normalize_raw_record("not a dict") is None

    # 2. Raw malformed dictionary from provider -> handled safely
    raw_messy = {
        "title": "Valid Paper Title",
        "authors": ["", "  Real Author  ", None, {"name": "Author Two"}],
        "year": "not-a-number",
        "publication_date": "Published on 2021-03-10",
        "doi": "not-a-real-doi",
        "externalIds": {"DOI": "https://doi.org/10.1016/j.valid.2021"},
    }
    doc = normalize_raw_record(raw_messy, source="semantic_scholar")
    assert doc is not None
    assert doc.title == "Valid Paper Title"
    assert "Real Author" in doc.authors
    assert "Author Two" in doc.authors
    assert doc.publication_year == 2021
    assert doc.doi == "10.1016/j.valid.2021"

    # 3. SourcePaper with string year and messy authors -> cleaned gracefully
    p_messy = SourcePaper(
        source="semantic_scholar",
        source_id="s2_9",
        title="Another Title",
        authors=["", "  Bob Jones  "],
        year="2022",  # type: ignore
        publication_date="2022-01-01",
        doi="invalid",
        raw_metadata="not a dict",  # type: ignore
    )
    doc2 = normalize_source_paper(p_messy)
    assert doc2 is not None
    assert doc2.title == "Another Title"
    assert doc2.authors == ["Bob Jones"]
    assert doc2.publication_year == 2022
    assert doc2.doi is None



# ============================================================================
# 2. SCHEMA VALIDATION
# ============================================================================

def test_research_document_schema_validation() -> None:
    doc = ResearchDocument(
        internal_id="doc_test123",
        source="openalex",
        source_id="W100",
        title="Schema Validation Paper",
        abstract="Testing schema constraints.",
        authors=["Author One"],
        publication_year=2024,
        citation_count=10,
        references=["W1", "W2"],
        citations=["W3"],
        provenance=[
            ProvenanceRecord(
                source="openalex",
                source_id="W100",
                original_url="https://openalex.org/W100",
            )
        ],
    )
    # Serialize to JSON and reload
    json_data = doc.model_dump_json()
    reloaded = ResearchDocument.model_validate_json(json_data)
    assert reloaded.internal_id == "doc_test123"
    assert reloaded.references == ["W1", "W2"]
    assert len(reloaded.provenance) == 1


# ============================================================================
# 3. DEDUPLICATION IN STRICT PRIORITY ORDER
# ============================================================================

def test_deduplicate_order_1_doi() -> None:
    """Rule 1: Matching DOI must merge records even if titles differ slightly."""
    doc1 = ResearchDocument(
        internal_id="doc_1",
        source="semantic_scholar",
        source_id="s2_a",
        title="Attention is all you need",
        doi="10.1000/182",
        citation_count=50000,
        authors=["Vaswani"],
    )
    doc2 = ResearchDocument(
        internal_id="doc_2",
        source="openalex",
        source_id="W_b",
        title="Attention Is All You Need (Official)",
        doi="10.1000/182",
        citation_count=60000,
        authors=["Ashish Vaswani"],
    )

    assert is_duplicate(doc1, doc2)
    merged = deduplicate_documents([doc1, doc2])
    assert len(merged) == 1
    assert merged[0].citation_count == 60000


def test_deduplicate_order_2_provider_id() -> None:
    """Rule 2: Matching provider identifier (source + source_id) must merge."""
    doc1 = ResearchDocument(
        internal_id="doc_1",
        source="openalex",
        source_id="W12345",
        title="Title from First Ingestion",
    )
    doc2 = ResearchDocument(
        internal_id="doc_2",
        source="openalex",
        source_id="W12345",
        title="Updated Title from Second Ingestion",
    )

    assert is_duplicate(doc1, doc2)
    merged = deduplicate_documents([doc1, doc2])
    assert len(merged) == 1


def test_deduplicate_order_3_exact_normalized_title() -> None:
    """Rule 3: Normalized exact title (without punctuation/case) must merge."""
    doc1 = ResearchDocument(
        internal_id="doc_1",
        source="semantic_scholar",
        source_id="s2_1",
        title="Deep Learning: A Comprehensive Survey!",
    )
    doc2 = ResearchDocument(
        internal_id="doc_2",
        source="openalex",
        source_id="W_2",
        title="deep learning a comprehensive survey",
    )

    assert is_duplicate(doc1, doc2)
    merged = deduplicate_documents([doc1, doc2])
    assert len(merged) == 1


def test_deduplicate_order_4_title_author_year_similarity() -> None:
    """Rule 4: High title similarity + author overlap + year compatibility must merge."""
    doc1 = ResearchDocument(
        internal_id="doc_1",
        source="semantic_scholar",
        source_id="s2_1",
        title="Scalable Vision Transformer Models in Medical Imaging",
        authors=["Jane Doe", "Bob Jones"],
        publication_year=2023,
    )
    doc2 = ResearchDocument(
        internal_id="doc_2",
        source="openalex",
        source_id="W_2",
        title="Scalable Vision Transformers in Medical Imaging",
        authors=["J. Doe"],
        publication_year=2023,
    )

    assert is_duplicate(doc1, doc2)
    merged = deduplicate_documents([doc1, doc2])
    assert len(merged) == 1


def test_deduplicate_distinct_papers() -> None:
    """Distinct papers must NOT be merged."""
    doc1 = ResearchDocument(
        internal_id="doc_1",
        source="semantic_scholar",
        source_id="s2_1",
        title="Convolutional Networks for Audio Recognition",
        doi="10.1000/audio1",
        publication_year=2020,
    )
    doc2 = ResearchDocument(
        internal_id="doc_2",
        source="openalex",
        source_id="W_2",
        title="Quantum Graph Neural Networks for Molecule Generation",
        doi="10.1000/quantum2",
        publication_year=2024,
    )

    assert not is_duplicate(doc1, doc2)
    merged = deduplicate_documents([doc1, doc2])
    assert len(merged) == 2


# ============================================================================
# 4. PROVENANCE PRESERVATION
# ============================================================================

def test_provenance_preservation_on_merge() -> None:
    prov1 = ProvenanceRecord(
        source="semantic_scholar",
        source_id="s2_99",
        original_url="https://semanticscholar.org/paper/s2_99",
        raw_metadata={"raw_key_1": "val1"},
    )
    doc1 = ResearchDocument(
        internal_id="doc_s2",
        source="semantic_scholar",
        source_id="s2_99",
        title="A Survey on Retrieval Augmented Generation",
        doi="10.1000/rag",
        citation_count=50,
        fields_of_study=["Natural Language Processing"],
        provenance=[prov1],
        raw_source_metadata={"semantic_scholar": {"raw_key_1": "val1"}},
    )

    prov2 = ProvenanceRecord(
        source="openalex",
        source_id="W_99",
        original_url="https://openalex.org/W_99",
        raw_metadata={"raw_key_2": "val2"},
    )
    doc2 = ResearchDocument(
        internal_id="doc_oa",
        source="openalex",
        source_id="W_99",
        title="A Survey on Retrieval-Augmented Generation (RAG)",
        doi="10.1000/rag",
        citation_count=80,
        concepts=["Information retrieval"],
        provenance=[prov2],
        raw_source_metadata={"openalex": {"raw_key_2": "val2"}},
    )

    merged = merge_documents(doc1, doc2)

    # Verify provenance retains both records
    assert len(merged.provenance) == 2
    sources = {p.source for p in merged.provenance}
    assert sources == {"semantic_scholar", "openalex"}

    # Consolidated source
    assert "semantic_scholar" in merged.source
    assert "openalex" in merged.source

    # Citation count is max
    assert merged.citation_count == 80

    # Concepts and fields merged
    assert "Natural Language Processing" in merged.fields_of_study
    assert "Information retrieval" in merged.concepts

    # Raw metadata preserved
    assert "semantic_scholar" in merged.raw_source_metadata
    assert "openalex" in merged.raw_source_metadata


# ============================================================================
# 5. CACHE HIT / MISS & TTL
# ============================================================================

def test_cache_key_generation() -> None:
    key1 = generate_cache_key("openalex", "deep learning", filters={"year": 2023}, pagination={"limit": 10})
    key2 = generate_cache_key("OPENALEX", "  deep   learning  ", filters={"year": 2023}, pagination={"limit": 10})
    assert key1 == key2


def test_cache_hit_miss_and_expiration() -> None:
    async def _run() -> None:
        cache = InMemoryCache(default_ttl_seconds=1)

        key = generate_cache_key("test_provider", "transformers", pagination={"limit": 5})

        # Cache miss
        assert await cache.get(key) is None

        # Cache set & hit
        sample_data = [{"id": 1, "name": "test"}]
        await cache.set(key, sample_data, ttl_seconds=1)
        cached = await cache.get(key)
        assert cached == sample_data

        # Cache expiration with short TTL
        await cache.set("expiring_key", "value", ttl_seconds=0.01)
        await asyncio.sleep(0.03)
        assert await cache.get("expiring_key") is None

        # Delete & clear
        await cache.set("k1", "v1")
        await cache.delete("k1")
        assert await cache.get("k1") is None

        await cache.set("k2", "v2")
        await cache.clear()
        assert await cache.get("k2") is None

    asyncio.run(_run())
