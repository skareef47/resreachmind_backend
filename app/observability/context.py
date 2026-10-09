"""
Context variable management for request tracing in ResearchMind AI.
"""

from contextvars import ContextVar
from typing import Optional

_request_id_ctx: ContextVar[Optional[str]] = ContextVar("current_request_id", default=None)


def set_current_request_id(request_id: str) -> None:
    """Set the active trace request ID for the current async task context."""
    _request_id_ctx.set(request_id)


def get_current_request_id() -> Optional[str]:
    """Retrieve the active trace request ID from context if set."""
    return _request_id_ctx.get()
