import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx

from app.schemas.source import SourcePaper

logger = logging.getLogger("researchmind.sources")

SENSITIVE_HEADER_KEYS = {"x-api-key", "authorization", "api-key", "api_key", "token"}
SENSITIVE_PARAM_KEYS = {"api_key", "apikey", "key", "token", "secret"}


def sanitize_headers(headers: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Return headers with sensitive credentials redacted for logging."""
    if not headers:
        return {}
    sanitized = {}
    for k, v in headers.items():
        if k.lower() in SENSITIVE_HEADER_KEYS:
            sanitized[k] = "[REDACTED]"
        else:
            sanitized[k] = v
    return sanitized


def sanitize_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return query params with sensitive credentials redacted for logging."""
    if not params:
        return {}
    sanitized = {}
    for k, v in params.items():
        if k.lower() in SENSITIVE_PARAM_KEYS:
            sanitized[k] = "[REDACTED]"
        else:
            sanitized[k] = v
    return sanitized


class SourceError(Exception):
    """Base exception for scholarly source errors."""

    def __init__(self, message: str, source: str) -> None:
        self.message = message
        self.source = source
        super().__init__(f"[{source}] {message}")


class SourceTimeoutError(SourceError):
    """Raised when an upstream source request exceeds the timeout threshold."""
    pass


class SourceRateLimitError(SourceError):
    """Raised when an upstream source returns 429 Too Many Requests and retries fail."""

    def __init__(self, message: str, source: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message, source)
        self.retry_after = retry_after


class SourceHttpError(SourceError):
    """Raised when an upstream source returns a non-200 HTTP response."""

    def __init__(self, message: str, source: str, status_code: int) -> None:
        super().__init__(message, source)
        self.status_code = status_code


class SourceMalformedResponseError(SourceError):
    """Raised when an upstream source returns an invalid or non-JSON payload."""
    pass


class BaseSourceAdapter(ABC):
    """Abstract base adapter for scholarly search data providers."""

    source_name: str = "base"

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily initialize or return the HTTP async client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
            self._owns_client = True
        return self._client

    async def close(self) -> None:
        """Close internal HTTP client if owned."""
        if self._owns_client and self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> "BaseSourceAdapter":
        await self._get_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    def _parse_retry_after(self, response: httpx.Response, attempt: int) -> float:
        """Calculate wait time from Retry-After header or exponential backoff."""
        retry_header = response.headers.get("Retry-After")
        if retry_header:
            try:
                seconds = float(retry_header)
                if 0 < seconds <= 30:
                    return seconds
            except ValueError:
                pass
        return self.backoff_factor * (2 ** attempt)

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Execute HTTP request with timeout, 429 rate-limit handling, and exponential backoff."""
        client = await self._get_client()
        last_error: Optional[Exception] = None

        safe_headers = sanitize_headers(headers)
        safe_params = sanitize_params(params)
        logger.debug(
            "Requesting [%s] %s %s with params=%s, headers=%s",
            self.source_name,
            method,
            url,
            safe_params,
            safe_headers,
        )

        for attempt in range(self.max_retries + 1):
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=headers,
                )

                if response.status_code == 200:
                    try:
                        return response.json()
                    except Exception as json_err:
                        raise SourceMalformedResponseError(
                            f"Failed to parse JSON response: {json_err}",
                            self.source_name,
                        ) from json_err

                if response.status_code == 429:
                    wait_seconds = self._parse_retry_after(response, attempt)
                    logger.warning(
                        "[%s] Rate limited (429) on attempt %d/%d. Waiting %.2fs before retry.",
                        self.source_name,
                        attempt + 1,
                        self.max_retries + 1,
                        wait_seconds,
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(wait_seconds)
                        continue
                    raise SourceRateLimitError(
                        f"Rate limit exceeded after {self.max_retries + 1} attempts",
                        self.source_name,
                        retry_after=wait_seconds,
                    )

                if 500 <= response.status_code < 600:
                    wait_seconds = self.backoff_factor * (2 ** attempt)
                    logger.warning(
                        "[%s] Upstream server error %d on attempt %d/%d. Waiting %.2fs.",
                        self.source_name,
                        response.status_code,
                        attempt + 1,
                        self.max_retries + 1,
                        wait_seconds,
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(wait_seconds)
                        continue
                    raise SourceHttpError(
                        f"Server error {response.status_code}",
                        self.source_name,
                        status_code=response.status_code,
                    )

                # Client error (4xx other than 429)
                raise SourceHttpError(
                    f"HTTP client error {response.status_code}: {response.text[:200]}",
                    self.source_name,
                    status_code=response.status_code,
                )

            except httpx.TimeoutException as timeout_err:
                last_error = timeout_err
                wait_seconds = self.backoff_factor * (2 ** attempt)
                logger.warning(
                    "[%s] Request timed out on attempt %d/%d. Waiting %.2fs.",
                    self.source_name,
                    attempt + 1,
                    self.max_retries + 1,
                    wait_seconds,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(wait_seconds)
                    continue
                raise SourceTimeoutError(
                    f"Request timed out after {self.max_retries + 1} attempts",
                    self.source_name,
                ) from timeout_err

            except httpx.RequestError as req_err:
                last_error = req_err
                wait_seconds = self.backoff_factor * (2 ** attempt)
                logger.warning(
                    "[%s] Network error on attempt %d/%d: %s. Waiting %.2fs.",
                    self.source_name,
                    attempt + 1,
                    self.max_retries + 1,
                    req_err,
                    wait_seconds,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(wait_seconds)
                    continue
                raise SourceHttpError(
                    f"Network connection failure: {req_err}",
                    self.source_name,
                    status_code=0,
                ) from req_err

        if last_error:
            raise SourceError(f"Request failed: {last_error}", self.source_name)
        raise SourceError("Request failed unexpectedly", self.source_name)

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[SourcePaper]:
        """Search upstream provider and return provider-neutral intermediate paper records."""
        pass
