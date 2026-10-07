"""
Database package for ResearchMind AI (Phase 9).
"""

from app.db.models import (
    Base,
    OptimizationResultDB,
    ResearchDocumentDB,
    ResearchRoadmapDB,
    SavedPaperDB,
    UserFeedbackDB,
    UserPreferencesDB,
    UtilityRunDB,
)
from app.db.repository import DatabaseRepository
from app.db.session import (
    check_db_health,
    get_db_session,
    get_engine,
    get_session_factory,
    init_db,
)

__all__ = [
    "Base",
    "ResearchDocumentDB",
    "SavedPaperDB",
    "UserFeedbackDB",
    "UserPreferencesDB",
    "UtilityRunDB",
    "OptimizationResultDB",
    "ResearchRoadmapDB",
    "DatabaseRepository",
    "get_engine",
    "get_session_factory",
    "get_db_session",
    "init_db",
    "check_db_health",
]
