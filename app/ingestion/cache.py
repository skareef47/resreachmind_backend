import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app.schemas.source import SourcePaper


def generate_cache_key(
    provider: str,
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    pagination: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a deterministic cache key for provider queries, filters, and pagination."""
    normalized_provider = provider.strip().lower()
    normalized_query = " ".join(query.strip().lower().split())

    payload = {
        "provider": normalized_provider,
        "query": normalized_query,
        "filters": filters or {},
        "pagination": pagination or {},
    }
    encoded = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"cache:{normalized_provider}:{digest}"


class BaseCache(ABC):
    """Abstract interface for query result caching."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache if present and unexpired."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Store item in cache with optional TTL in seconds."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Evict item from cache."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached items."""
        pass


class InMemoryCache(BaseCache):
    """In-memory thread-safe TTL cache implementation."""

    def __init__(self, default_ttl_seconds: int = 3600) -> None:
        self.default_ttl = default_ttl_seconds
        self._store: Dict[str, tuple[Any, Optional[float]]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key not in self._store:
                return None
            value, expires_at = self._store[key]
            if expires_at is not None and time.time() > expires_at:
                del self._store[key]
                return None
            return value

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expires_at = (time.time() + ttl) if ttl > 0 else None
        async with self._lock:
            self._store[key] = (value, expires_at)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    async def size(self) -> int:
        async with self._lock:
            now = time.time()
            return sum(
                1 for _, (_, expires_at) in self._store.items()
                if expires_at is None or expires_at > now
            )


# Global singleton cache instance for application lifecycle
_global_cache = InMemoryCache()


def get_cache() -> InMemoryCache:
    """Dependency provider returning global in-memory cache instance."""
    return _global_cache
