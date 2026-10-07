import difflib
import re
from typing import List, Optional, Set

from app.schemas.paper import ProvenanceRecord, ResearchDocument

ALPHANUM_REGEX = re.compile(r"[^a-z0-9\s]")


def normalize_title_for_comparison(title: Optional[str]) -> str:
    """Normalize title for exact and fuzzy matching."""
    if not title:
        return ""
    lowered = title.lower()
    cleaned = ALPHANUM_REGEX.sub(" ", lowered)
    return " ".join(cleaned.split())


def extract_last_names(authors: List[str]) -> Set[str]:
    """Extract lowercase author surnames for similarity comparison."""
    last_names = set()
    for author in authors:
        parts = author.strip().split()
        if parts:
            last_names.add(parts[-1].lower())
    return last_names


def title_similarity_ratio(title1: str, title2: str) -> float:
    """Compute token-level Jaccard similarity and character sequence ratio."""
    t1 = normalize_title_for_comparison(title1)
    t2 = normalize_title_for_comparison(title2)
    if not t1 or not t2:
        return 0.0
    if t1 == t2:
        return 1.0

    # Token Jaccard
    tokens1 = set(t1.split())
    tokens2 = set(t2.split())
    if not tokens1 or not tokens2:
        return 0.0
    jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2)

    # Sequence Matcher
    seq_ratio = difflib.SequenceMatcher(None, t1, t2).ratio()
    return max(jaccard, seq_ratio)


def semantic_similarity_safeguard(doc1: ResearchDocument, doc2: ResearchDocument) -> bool:
    """Safeguard semantic similarity placeholder for Phase 3 before Phase 4 SBERT."""
    # N-gram overlap safeguard for titles + abstract snippets if both present
    if not doc1.abstract or not doc2.abstract:
        return False
    words1 = set(normalize_title_for_comparison(doc1.abstract[:200]).split())
    words2 = set(normalize_title_for_comparison(doc2.abstract[:200]).split())
    if len(words1) >= 10 and len(words2) >= 10:
        overlap = len(words1 & words2) / min(len(words1), len(words2))
        return overlap >= 0.85
    return False


def is_duplicate(doc1: ResearchDocument, doc2: ResearchDocument) -> bool:
    """Check if two ResearchDocuments represent the same work in strict priority order.

    Deduplication Order:
    1. DOI
    2. Provider Identifier
    3. Normalized exact title
    4. Title + Author + Year similarity
    5. Semantic similarity safeguard
    """
    # 1. DOI Matching
    if doc1.doi and doc2.doi:
        if doc1.doi.strip().lower() == doc2.doi.strip().lower():
            return True

    # 2. Provider Identifier Matching
    if doc1.source == doc2.source and doc1.source_id and doc2.source_id:
        if doc1.source_id.strip() == doc2.source_id.strip():
            return True

    # 3. Normalized Exact Title Matching
    norm_t1 = normalize_title_for_comparison(doc1.title)
    norm_t2 = normalize_title_for_comparison(doc2.title)
    if norm_t1 and norm_t2 and len(norm_t1) > 8 and norm_t1 == norm_t2:
        return True

    # 4. Title + Author + Year Similarity Matching
    sim = title_similarity_ratio(doc1.title, doc2.title)
    if sim >= 0.85:
        # Check year compatibility
        year_match = True
        if doc1.publication_year and doc2.publication_year:
            year_match = abs(doc1.publication_year - doc2.publication_year) <= 1

        # Check author overlap
        authors1 = extract_last_names(doc1.authors)
        authors2 = extract_last_names(doc2.authors)
        author_match = True
        if authors1 and authors2:
            author_match = bool(authors1 & authors2)

        if year_match and author_match:
            return True

    # 5. Semantic similarity safeguard
    if semantic_similarity_safeguard(doc1, doc2):
        return True

    return False


def merge_documents(doc1: ResearchDocument, doc2: ResearchDocument) -> ResearchDocument:
    """Merge two duplicate ResearchDocuments, preserving provenance and combining metadata."""
    # 1. Combined provenance (audit trail preserved)
    merged_provenance: List[ProvenanceRecord] = list(doc1.provenance)
    seen_prov = {(p.source, p.source_id) for p in merged_provenance}
    for p in doc2.provenance:
        if (p.source, p.source_id) not in seen_prov:
            merged_provenance.append(p)
            seen_prov.add((p.source, p.source_id))

    # 2. Source consolidated
    sources = sorted(set(doc1.source.split(",") + doc2.source.split(",")))
    merged_source = ",".join(s.strip() for s in sources if s.strip())

    # 3. Title (prefer longer or cleaner)
    merged_title = doc1.title if len(doc1.title) >= len(doc2.title) else doc2.title

    # 4. Abstract (prefer non-null and longer)
    merged_abstract = doc1.abstract
    if not merged_abstract or (doc2.abstract and len(doc2.abstract) > len(merged_abstract)):
        merged_abstract = doc2.abstract

    # 5. Authors (union preserving order)
    merged_authors = list(doc1.authors)
    seen_authors = {a.lower() for a in merged_authors}
    for a in doc2.authors:
        if a.lower() not in seen_authors:
            merged_authors.append(a)
            seen_authors.add(a.lower())

    # 6. Year & Date
    merged_year = doc1.publication_year or doc2.publication_year
    merged_date = doc1.publication_date or doc2.publication_date

    # 7. DOI & URL
    merged_doi = doc1.doi or doc2.doi
    merged_url = doc1.url or doc2.url

    # 8. Citation Count (take maximum reported)
    merged_citations = max(doc1.citation_count or 0, doc2.citation_count or 0)

    # 9. Venue
    merged_venue = doc1.venue or doc2.venue

    # 10. Fields of study & Concepts (unions)
    merged_fields = list(dict.fromkeys(doc1.fields_of_study + doc2.fields_of_study))
    merged_concepts = list(dict.fromkeys(doc1.concepts + doc2.concepts))

    # 11. Datasets, Methods, Models
    merged_datasets = list(dict.fromkeys(doc1.datasets + doc2.datasets))
    merged_methods = list(dict.fromkeys(doc1.methods + doc2.methods))
    merged_models = list(dict.fromkeys(doc1.models + doc2.models))

    # 12. Code URL, Open Access, License
    merged_code = doc1.code_url or doc2.code_url
    merged_oa = (doc1.open_access or doc2.open_access) if (doc1.open_access is not None or doc2.open_access is not None) else None
    merged_license = doc1.license or doc2.license

    # 13. Raw metadata merged
    merged_raw = dict(doc1.raw_source_metadata)
    merged_raw.update(doc2.raw_source_metadata)

    return ResearchDocument(
        internal_id=doc1.internal_id,
        source=merged_source,
        source_id=doc1.source_id,
        title=merged_title,
        abstract=merged_abstract,
        authors=merged_authors,
        publication_year=merged_year,
        publication_date=merged_date,
        doi=merged_doi,
        url=merged_url,
        citation_count=merged_citations,
        references=list(dict.fromkeys(doc1.references + doc2.references)),
        citations=list(dict.fromkeys(doc1.citations + doc2.citations)),
        venue=merged_venue,
        fields_of_study=merged_fields,
        concepts=merged_concepts,
        datasets=merged_datasets,
        methods=merged_methods,
        models=merged_models,
        code_url=merged_code,
        open_access=merged_oa,
        license=merged_license,
        provenance=merged_provenance,
        raw_source_metadata=merged_raw,
        retrieval_timestamp=doc1.retrieval_timestamp,
    )


def deduplicate_documents(documents: List[ResearchDocument]) -> List[ResearchDocument]:
    """Deduplicate an input list of ResearchDocuments, merging duplicates and preserving provenance."""
    canonical: List[ResearchDocument] = []

    for doc in documents:
        matched_idx: Optional[int] = None
        for i, existing in enumerate(canonical):
            if is_duplicate(doc, existing):
                matched_idx = i
                break

        if matched_idx is not None:
            # Merge with existing canonical record
            canonical[matched_idx] = merge_documents(canonical[matched_idx], doc)
        else:
            canonical.append(doc)

    return canonical
