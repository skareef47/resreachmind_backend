"""Ingestion pipeline package: normalization, deduplication, and caching."""

from app.ingestion.cache import BaseCache, InMemoryCache, generate_cache_key, get_cache
from app.ingestion.deduplicate import (
    deduplicate_documents,
    is_duplicate,
    merge_documents,
    title_similarity_ratio,
)
from app.ingestion.normalize import (
    clean_doi,
    clean_title,
    normalize_raw_record,
    normalize_source_paper,
)

__all__ = [
    "clean_title",
    "clean_doi",
    "normalize_source_paper",
    "normalize_raw_record",
    "is_duplicate",
    "merge_documents",
    "deduplicate_documents",
    "title_similarity_ratio",
    "BaseCache",
    "InMemoryCache",
    "generate_cache_key",
    "get_cache",
]
