"""
Secret Redaction Engine for ResearchMind AI.

Ensures that sensitive information (API keys, authorization tokens, passwords,
private credentials) is NEVER logged or exposed through observability endpoints.
"""

import logging
import re
from typing import Any, Dict, List, Set, Union

# Common parameter and header names containing secrets
SENSITIVE_FIELD_NAMES: Set[str] = {
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "authorization",
    "bearer",
    "client_secret",
    "credentials",
    "db_password",
    "key",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "token",
}

# Regex patterns matching key-value pairs or tokens in strings
REDACTION_PATTERNS: List[re.Pattern] = [
    # Key-value pairs: e.g. api_key="sk-12345" or db_password: secret123
    re.compile(
        r"(?i)\b([a-zA-Z0-9_\-\.]*(?:api[_-]?key|access[_-]?token|auth[_-]?token|authorization|bearer|client[_-]?secret|credentials|password|private[_-]?key|secret|token)[a-zA-Z0-9_\-\.]*)\s*([:=])\s*['\"]?([^\s'\",;&]{4,})['\"]?",
        re.IGNORECASE,
    ),
    # Authorization header style: Bearer ey...
    re.compile(r"(?i)\b(Bearer\s+)[a-zA-Z0-9_\-\.]{8,}", re.IGNORECASE),
    # OpenAI/SemanticScholar style secret keys: sk-...
    re.compile(r"\b(sk-[a-zA-Z0-9]{12,})\b"),
    # Generic long hex or alphanumeric tokens in query strings: ?key=abcd1234efgh5678
    re.compile(r"(?i)([?&](?:key|api_key|token|secret)=)[a-zA-Z0-9_\-\.]{6,}", re.IGNORECASE),
]


def redact_sensitive_text(text: str) -> str:
    """
    Scans a string and masks any detected secrets, API keys, or credentials
    with [REDACTED].
    """
    if not text:
        return text

    redacted = text
    # Pattern 1: key-value matches
    redacted = REDACTION_PATTERNS[0].sub(r"\1\2[REDACTED]", redacted)
    # Pattern 2: Bearer tokens
    redacted = REDACTION_PATTERNS[1].sub(r"\1[REDACTED]", redacted)
    # Pattern 3: sk- keys
    redacted = REDACTION_PATTERNS[2].sub(r"[REDACTED]", redacted)
    # Pattern 4: query string keys
    redacted = REDACTION_PATTERNS[3].sub(r"\1[REDACTED]", redacted)

    return redacted


def redact_sensitive_dict(data: Any) -> Any:
    """
    Recursively scans dicts, lists, and primitives, masking sensitive keys.
    """
    if isinstance(data, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in data.items():
            k_lower = str(k).lower().replace("-", "_").replace(" ", "_")
            if k_lower in SENSITIVE_FIELD_NAMES or any(s in k_lower for s in ["api_key", "password", "secret", "token"]):
                cleaned[k] = "[REDACTED]"
            elif isinstance(v, (dict, list)):
                cleaned[k] = redact_sensitive_dict(v)
            elif isinstance(v, str):
                cleaned[k] = redact_sensitive_text(v)
            else:
                cleaned[k] = v
        return cleaned
    elif isinstance(data, list):
        return [redact_sensitive_dict(item) for item in data]
    elif isinstance(data, str):
        return redact_sensitive_text(data)
    else:
        return data


class RedactingLogFilter(logging.Filter):
    """
    Python logging filter that automatically intercepts and redacts sensitive
    credentials from log messages and formatting arguments.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_sensitive_text(record.msg)

        if record.args:
            if isinstance(record.args, dict):
                record.args = redact_sensitive_dict(record.args)
            elif isinstance(record.args, (list, tuple)):
                cleaned_args = []
                for arg in record.args:
                    if isinstance(arg, str):
                        cleaned_args.append(redact_sensitive_text(arg))
                    elif isinstance(arg, (dict, list)):
                        cleaned_args.append(redact_sensitive_dict(arg))
                    else:
                        cleaned_args.append(arg)
                record.args = tuple(cleaned_args)

        return True
