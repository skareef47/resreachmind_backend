"""Scholarly source adapters package."""

from app.sources.base import (
    BaseSourceAdapter,
    SourceError,
    SourceHttpError,
    SourceMalformedResponseError,
    SourceRateLimitError,
    SourceTimeoutError,
    sanitize_headers,
)
from app.sources.openalex import OpenAlexAdapter
from app.sources.semantic_scholar import SemanticScholarAdapter

__all__ = [
    "BaseSourceAdapter",
    "SourceError",
    "SourceTimeoutError",
    "SourceRateLimitError",
    "SourceHttpError",
    "SourceMalformedResponseError",
    "sanitize_headers",
    "SemanticScholarAdapter",
    "OpenAlexAdapter",
]
