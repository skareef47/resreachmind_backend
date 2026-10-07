import html
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.schemas.paper import ProvenanceRecord, ResearchDocument
from app.schemas.source import SourcePaper

DOI_PREFIX_REGEX = re.compile(r"^(https?://(dx\.)?doi\.org/|doi:)", re.IGNORECASE)
YEAR_REGEX = re.compile(r"\b(19\d\d|20\d\d)\b")
WHITESPACE_REGEX = re.compile(r"\s+")
CODE_REPO_REGEX = re.compile(r"https?://(github\.com|gitlab\.com|huggingface\.co)/[a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-]+", re.IGNORECASE)


def clean_title(title: Optional[str]) -> Optional[str]:
    """Clean and normalize paper title string."""
    if not title or not isinstance(title, str):
        return None
    unescaped = html.unescape(title)
    cleaned = WHITESPACE_REGEX.sub(" ", unescaped).strip()
    return cleaned if cleaned else None


def clean_doi(doi: Optional[str]) -> Optional[str]:
    """Normalize DOI to lowercase identifier without URL prefixes."""
    if not doi or not isinstance(doi, str):
        return None
    trimmed = doi.strip()
    stripped = DOI_PREFIX_REGEX.sub("", trimmed).strip().lower()
    if "/" in stripped and len(stripped) > 4:
        return stripped
    return None


def clean_authors(authors: Optional[List[Any]]) -> List[str]:
    """Clean and deduplicate author names while preserving original order."""
    if not authors or not isinstance(authors, list):
        return []
    cleaned_list: List[str] = []
    seen = set()
    for item in authors:
        name: Optional[str] = None
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            name = item.get("name") or item.get("display_name")
            if isinstance(name, str):
                name = name.strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            cleaned_list.append(name)
    return cleaned_list


def clean_year(year: Optional[Any], date_str: Optional[str]) -> Optional[int]:
    """Extract and validate 4-digit publication year."""
    if isinstance(year, int) and 1800 <= year <= 2100:
        return year
    if isinstance(year, str) and year.isdigit():
        val = int(year)
        if 1800 <= val <= 2100:
            return val
    if date_str and isinstance(date_str, str):
        match = YEAR_REGEX.search(date_str)
        if match:
            return int(match.group(1))
    return None


def extract_code_url(raw_metadata: Dict[str, Any], url: Optional[str]) -> Optional[str]:
    """Detect and extract source code repository URL if present in metadata."""
    candidates = []
    if url:
        candidates.append(url)

    # Check external IDs or links
    if isinstance(raw_metadata, dict):
        for key in ("codeUrl", "code_url", "repository", "github"):
            val = raw_metadata.get(key)
            if isinstance(val, str):
                candidates.append(val)
        open_access_pdf = raw_metadata.get("openAccessPdf")
        if isinstance(open_access_pdf, dict) and open_access_pdf.get("url"):
            candidates.append(open_access_pdf["url"])

    for candidate in candidates:
        match = CODE_REPO_REGEX.search(candidate)
        if match:
            return match.group(0)
    return None


def extract_open_access_and_license(raw: Dict[str, Any], source: str) -> Tuple[Optional[bool], Optional[str]]:
    """Extract open access status and license identifier from raw source metadata."""
    if not isinstance(raw, dict):
        return None, None

    is_oa: Optional[bool] = None
    license_str: Optional[str] = None

    if raw.get("open_access") is True or raw.get("isOpenAccess") is True or raw.get("is_oa") is True:
        is_oa = True

    if source == "openalex":
        oa_obj = raw.get("open_access")
        if isinstance(oa_obj, dict):
            is_oa = oa_obj.get("is_oa")
        elif isinstance(oa_obj, bool):
            is_oa = oa_obj
        primary_loc = raw.get("primary_location")
        if isinstance(primary_loc, dict):
            license_str = primary_loc.get("license")
    elif source == "semantic_scholar":
        if is_oa is None:
            is_oa = raw.get("isOpenAccess")
        if not is_oa and raw.get("openAccessPdf"):
            is_oa = True

    return is_oa, license_str


def extract_fields_and_concepts(raw: Dict[str, Any], source: str) -> Tuple[List[str], List[str]]:
    """Extract fields of study and ontology concepts from provider payload."""
    fields_of_study: List[str] = []
    concepts: List[str] = []

    if not isinstance(raw, dict):
        return fields_of_study, concepts

    if source == "openalex":
        raw_concepts = raw.get("concepts") or []
        for c in raw_concepts:
            if isinstance(c, dict) and c.get("display_name"):
                concepts.append(c["display_name"])
    elif source == "semantic_scholar":
        raw_fields = raw.get("fieldsOfStudy") or []
        for f in raw_fields:
            if isinstance(f, str) and f.strip():
                fields_of_study.append(f.strip())

    return fields_of_study, concepts


def generate_internal_id(
    doi: Optional[str],
    source: str,
    source_id: str,
    title: Optional[str],
) -> str:
    """Generate a deterministic UUID identifier for canonical document."""
    if doi:
        token = f"doi:{doi}"
    elif title:
        norm_title = re.sub(r"[^a-z0-9]", "", title.lower())
        token = f"title:{norm_title}"
    else:
        token = f"{source}:{source_id}"
    return f"doc_{uuid.uuid5(uuid.NAMESPACE_URL, token).hex}"


def normalize_source_paper(paper: SourcePaper) -> Optional[ResearchDocument]:
    """Normalize a provider SourcePaper into a canonical ResearchDocument."""
    title = clean_title(paper.title)
    if not title:
        return None

    doi = clean_doi(paper.doi)
    year = clean_year(paper.year, paper.publication_date)
    authors = clean_authors(paper.authors)
    abstract = paper.abstract.strip() if (paper.abstract and isinstance(paper.abstract, str)) else None

    raw = paper.raw_metadata if isinstance(paper.raw_metadata, dict) else {}
    code_url = extract_code_url(raw, paper.url)
    open_access, license_info = extract_open_access_and_license(raw, paper.source)
    fields_of_study, concepts = extract_fields_and_concepts(raw, paper.source)

    internal_id = generate_internal_id(doi, paper.source, paper.source_id, title)
    now_iso = datetime.now(timezone.utc).isoformat()

    provenance_entry = ProvenanceRecord(
        source=paper.source,
        source_id=paper.source_id,
        retrieved_at=now_iso,
        original_url=paper.url,
        raw_metadata=raw,
    )

    return ResearchDocument(
        internal_id=internal_id,
        source=paper.source,
        source_id=paper.source_id,
        title=title,
        abstract=abstract,
        authors=authors,
        publication_year=year,
        publication_date=paper.publication_date,
        doi=doi,
        url=paper.url,
        citation_count=paper.citation_count or 0,
        references=[],
        citations=[],
        venue=paper.venue,
        fields_of_study=fields_of_study,
        concepts=concepts,
        datasets=[],
        methods=[],
        models=[],
        code_url=code_url,
        open_access=open_access,
        license=license_info,
        provenance=[provenance_entry],
        raw_source_metadata={paper.source: raw} if raw else {},
        retrieval_timestamp=now_iso,
    )


def normalize_raw_record(
    raw: Any,
    source: str = "unknown",
) -> Optional[ResearchDocument]:
    """Normalize a raw, potentially malformed provider record payload into a ResearchDocument."""
    if not isinstance(raw, dict):
        return None

    raw_title = raw.get("title") or raw.get("display_name")
    title = clean_title(raw_title)
    if not title:
        return None

    source_id = str(raw.get("id") or raw.get("paperId") or raw.get("source_id") or "unknown")
    doi = clean_doi(raw.get("doi"))
    if not doi and isinstance(raw.get("externalIds"), dict):
        doi = clean_doi(raw["externalIds"].get("DOI"))

    raw_authors = raw.get("authors") or raw.get("authorships") or []
    authors = clean_authors(raw_authors)

    year = clean_year(
        raw.get("year") or raw.get("publication_year"),
        raw.get("publication_date") or raw.get("publicationDate") or raw.get("date"),
    )

    abstract = raw.get("abstract")
    if not isinstance(abstract, str):
        abstract = None

    url = raw.get("url") or (f"https://doi.org/{doi}" if doi else None)
    citations = raw.get("citation_count") or raw.get("citationCount") or raw.get("cited_by_count") or 0
    venue = raw.get("venue") or (
        raw.get("primary_location", {}).get("source", {}).get("display_name")
        if isinstance(raw.get("primary_location"), dict)
        else None
    )

    code_url = extract_code_url(raw, url)
    open_access, license_info = extract_open_access_and_license(raw, source)
    fields_of_study, concepts = extract_fields_and_concepts(raw, source)

    internal_id = generate_internal_id(doi, source, source_id, title)
    now_iso = datetime.now(timezone.utc).isoformat()

    provenance_entry = ProvenanceRecord(
        source=source,
        source_id=source_id,
        retrieved_at=now_iso,
        original_url=url,
        raw_metadata=raw,
    )

    return ResearchDocument(
        internal_id=internal_id,
        source=source,
        source_id=source_id,
        title=title,
        abstract=abstract,
        authors=authors,
        publication_year=year,
        publication_date=str(raw.get("publication_date") or raw.get("publicationDate") or "") or None,
        doi=doi,
        url=url,
        citation_count=citations if isinstance(citations, int) else 0,
        references=[],
        citations=[],
        venue=venue,
        fields_of_study=fields_of_study,
        concepts=concepts,
        datasets=[],
        methods=[],
        models=[],
        code_url=code_url,
        open_access=open_access,
        license=license_info,
        provenance=[provenance_entry],
        raw_source_metadata={source: raw},
        retrieval_timestamp=now_iso,
    )

