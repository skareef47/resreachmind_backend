"""
SQLAlchemy database models for ResearchMind AI (Phase 9).

Supports PostgreSQL (and SQLite fallback) for:
- ResearchDocument & Provenance
- Saved Papers
- User Feedback
- User Preferences
- Utility Runs & Audit Telemetry
- Optimization Results
- Research Roadmaps
"""

from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy declarative models."""
    pass


class ResearchDocumentDB(Base):
    """Persistent storage for canonical ResearchDocuments and provenance."""

    __tablename__ = "research_documents"

    internal_id = Column(String(100), primary_key=True, index=True)
    source = Column(String(100), nullable=False)
    source_id = Column(String(255), nullable=False, index=True)
    title = Column(Text, nullable=False, index=True)
    abstract = Column(Text, nullable=True)
    authors = Column(JSON, default=list)
    publication_year = Column(Integer, nullable=True, index=True)
    publication_date = Column(String(50), nullable=True)
    doi = Column(String(255), nullable=True, index=True)
    url = Column(Text, nullable=True)
    citation_count = Column(Integer, default=0, index=True)
    references = Column(JSON, default=list)
    citations = Column(JSON, default=list)
    venue = Column(String(255), nullable=True, index=True)
    fields_of_study = Column(JSON, default=list)
    concepts = Column(JSON, default=list)
    datasets = Column(JSON, default=list)
    methods = Column(JSON, default=list)
    models = Column(JSON, default=list)
    code_url = Column(Text, nullable=True)
    open_access = Column(Boolean, nullable=True)
    license = Column(String(100), nullable=True)
    provenance = Column(JSON, default=list)
    raw_source_metadata = Column(JSON, default=dict)
    retrieval_timestamp = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class SavedPaperDB(Base):
    """Saved papers bookmarked by users."""

    __tablename__ = "saved_papers"

    id = Column(String(100), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(100), ForeignKey("research_documents.internal_id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String(100), default="default_user", nullable=False, index=True)
    notes = Column(Text, nullable=True)
    tags = Column(JSON, default=list)
    saved_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    document = relationship("ResearchDocumentDB", backref="saved_entries", lazy="joined")


class UserFeedbackDB(Base):
    """User feedback signals for ranking refinement."""

    __tablename__ = "user_feedback"

    id = Column(String(100), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(100), nullable=False, index=True)
    feedback_type = Column(String(50), nullable=False)
    target_document_id = Column(String(100), nullable=True)
    rationale = Column(Text, nullable=True)
    user_id = Column(String(100), default="default_user", nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class UserPreferencesDB(Base):
    """Multi-dimensional user preference configurations."""

    __tablename__ = "user_preferences"

    user_id = Column(String(100), primary_key=True)
    preferences = Column(JSON, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class UtilityRunDB(Base):
    """Audit persistence for formal ranking runs."""

    __tablename__ = "utility_runs"

    run_id = Column(String(100), primary_key=True, index=True)
    weights = Column(JSON, nullable=False)
    preferences = Column(JSON, nullable=False)
    feedback_items = Column(JSON, nullable=False)
    constraints = Column(JSON, nullable=False)
    total_candidates = Column(Integer, nullable=False)
    admissible_count = Column(Integer, nullable=False)
    evaluations = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class OptimizationResultDB(Base):
    """Audit persistence for global optimization solver runs."""

    __tablename__ = "optimization_results"

    id = Column(String(100), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(String(100), index=True, nullable=True)
    objective_value = Column(Float, nullable=False)
    solver_used = Column(String(50), nullable=False)
    solver_status = Column(String(50), nullable=False)
    ranked_document_ids = Column(JSON, nullable=False)
    verification_report = Column(JSON, nullable=False)
    guarantee_scope = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ResearchRoadmapDB(Base):
    """Persistent storage for generated engineering roadmaps."""

    __tablename__ = "research_roadmaps"

    roadmap_id = Column(String(100), primary_key=True, index=True)
    target_query = Column(Text, nullable=False)
    title = Column(Text, nullable=False)
    stages = Column(JSON, nullable=False)
    supporting_papers = Column(JSON, default=list)
    total_duration = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
