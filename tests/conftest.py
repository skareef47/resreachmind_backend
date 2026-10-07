"""Pytest configuration and session fixtures for ResearchMind AI tests."""

import platform

# Ensure platform.uname cache is primed to avoid repeated Windows WMI queries
try:
    if hasattr(platform, "_uname_cache") and platform._uname_cache is None:
        platform.uname()
except Exception:
    pass
