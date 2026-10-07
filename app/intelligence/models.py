"""
Models for ResearchMind AI Phase 8 — Research Intelligence.

Provides schemas for:
1. Explainability: Component breakdown and narrative generated strictly from stored ranking inputs.
2. Buildability: 9-dimensional engineering feasibility scoring with concrete evidence.
3. Knowledge Graph: Multi-entity graph (Paper -> Dataset/Model/Method/Repository/Related Paper) with provenance.
4. Roadmap: 7-stage engineering implementation roadmap linked to supporting research and resources.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# =========================================================================
# 1. EXPLAINABILITY MODELS
# =========================================================================

class ComponentContribution(BaseModel):
    """Numerical and narrative breakdown for a single utility component."""

    name: str = Field(..., description="Component name (relevance, preference_match, etc.)")
    weight: float = Field(..., description="Configured weight in utility model")
    raw_score: float = Field(..., description="Evaluated score in [0.0, 1.0]")
    weighted_score: float = Field(..., description="weight * raw_score")
    percentage_contribution: float = Field(..., description="Percentage of total positive utility")
    description: str = Field(..., description="Concise explanation grounded in inputs")


class RankingExplanation(BaseModel):
    """
    Transparent ranking explanation generated ONLY from actual stored ranking inputs
    and run telemetry.
    """

    document_id: str = Field(..., description="Target document internal identifier")
    title: str = Field(..., description="Paper title")
    run_id: str = Field(..., description="Associated immutable ranking run identifier")
    rank: Optional[int] = Field(default=None, description="Achieved rank position in run")
    final_utility_score: float = Field(..., description="Overall scalar utility score U(R)")
    components: Dict[str, ComponentContribution] = Field(
        ..., description="Decomposed component contributions"
    )
    narrative_explanation: str = Field(
        ..., description="Complete human-readable narrative grounded in stored run inputs"
    )
    active_preferences_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Active user preference dimensions during run"
    )
    active_feedback_summary: List[str] = Field(
        default_factory=list, description="User feedback signals directly affecting this candidate"
    )
    applied_penalties_summary: Dict[str, float] = Field(
        default_factory=dict, description="Itemized soft constraint penalty deductions"
    )
    provenance_run_timestamp: str = Field(
        ..., description="Timestamp of ranking run execution"
    )


# =========================================================================
# 2. BUILDABILITY MODELS
# =========================================================================

class BuildabilityDimension(BaseModel):
    """Detailed score and factual evidence for a single buildability dimension."""

    name: str = Field(..., description="Dimension name")
    score: float = Field(..., ge=0.0, le=1.0, description="Evaluated dimension feasibility [0.0, 1.0]")
    evidence: List[str] = Field(default_factory=list, description="Concrete factual evidence items")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Assessment confidence")


class BuildabilityReport(BaseModel):
    """
    Rigorous engineering buildability evaluation across 9 core dimensions:
    - dataset availability
    - source-code availability
    - pretrained model availability
    - documentation
    - hardware requirements
    - dependency complexity
    - license/accessibility
    - reproducibility
    - implementation completeness
    """

    document_id: str = Field(..., description="Target document identifier")
    title: str = Field(..., description="Document title")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall weighted buildability score")
    reproducibility_tier: str = Field(..., description="Tier: High, Moderate, Low, or Blocked")
    dimensions: Dict[str, BuildabilityDimension] = Field(..., description="Breakdown of all 9 dimensions")
    key_strengths: List[str] = Field(default_factory=list, description="Top engineering enablers")
    key_risks_or_blockers: List[str] = Field(default_factory=list, description="Top engineering risks/barriers")
    evaluated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of evaluation",
    )


# =========================================================================
# 3. KNOWLEDGE GRAPH MODELS
# =========================================================================

class NodeType(str, Enum):
    PAPER = "paper"
    DATASET = "dataset"
    MODEL = "model"
    METHOD = "method"
    REPOSITORY = "repository"


class EdgeType(str, Enum):
    USES_DATASET = "uses_dataset"
    PRODUCES_DATASET = "produces_dataset"
    USES_MODEL = "uses_model"
    PROPOSES_MODEL = "proposes_model"
    USES_METHOD = "uses_method"
    PROPOSES_METHOD = "proposes_method"
    HAS_REPOSITORY = "has_repository"
    CITES_PAPER = "cites_paper"
    SIMILAR_TO_PAPER = "similar_to_paper"


class GraphProvenance(BaseModel):
    """Provenance tracking origin and confidence for nodes and edges."""

    source_document_id: str = Field(..., description="Origin document ID")
    source_provider: str = Field(..., description="Scholarly provider or extraction subsystem")
    extracted_field: str = Field(..., description="Source metadata field (datasets, code_url, etc.)")
    extracted_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")


class GraphNode(BaseModel):
    """Entity node in the scholarly research knowledge graph."""

    id: str = Field(..., description="Globally unique node ID (e.g. 'paper:doc_123', 'dataset:imagenet')")
    type: NodeType = Field(..., description="Entity category")
    label: str = Field(..., description="Human-readable entity name")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Entity attributes")
    provenance: GraphProvenance = Field(..., description="Data provenance")


class GraphEdge(BaseModel):
    """Directed semantic relationship edge in the knowledge graph."""

    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    type: EdgeType = Field(..., description="Relationship semantic category")
    label: str = Field(..., description="Relationship display label")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relationship metadata")
    provenance: GraphProvenance = Field(..., description="Relationship provenance")


class KnowledgeGraph(BaseModel):
    """Comprehensive multi-entity knowledge graph representation."""

    nodes: List[GraphNode] = Field(default_factory=list, description="Entity nodes")
    edges: List[GraphEdge] = Field(default_factory=list, description="Relationship edges")
    total_nodes: int = Field(..., description="Total node count")
    total_edges: int = Field(..., description="Total edge count")
    summary_by_type: Dict[str, int] = Field(default_factory=dict, description="Node counts by entity type")


# =========================================================================
# 4. ROADMAP MODELS
# =========================================================================

class RoadmapStageType(str, Enum):
    DATASET_PREPARATION = "dataset_preparation"
    BASELINE = "baseline"
    MODEL_TRAINING = "model_training"
    EVALUATION = "evaluation"
    BACKEND = "backend"
    FRONTEND_INTEGRATION = "frontend_integration"
    DEPLOYMENT = "deployment"


class SupportingResource(BaseModel):
    """Research reference or engineering asset linked to a roadmap milestone."""

    resource_type: str = Field(..., description="Type: paper, dataset, repository, model, method, docs")
    title: str = Field(..., description="Resource name or paper title")
    url: Optional[str] = Field(default=None, description="External URL if available")
    identifier: Optional[str] = Field(default=None, description="DOI, GitHub slug, HF repo ID")
    document_id: Optional[str] = Field(default=None, description="Internal ResearchDocument ID if linked")
    relevance_note: str = Field(..., description="How this resource supports the milestone")


class RoadmapMilestone(BaseModel):
    """Actionable development milestone in the 7-stage research roadmap."""

    id: str = Field(..., description="Unique milestone ID (e.g. 'ms_dataset_prep')")
    stage: RoadmapStageType = Field(..., description="Development stage")
    title: str = Field(..., description="Milestone title")
    description: str = Field(..., description="Actionable execution steps")
    prerequisites: List[str] = Field(default_factory=list, description="Required milestone IDs before starting")
    estimated_effort: str = Field(..., description="Estimated effort (e.g. '1-2 days', '1 week')")
    deliverables: List[str] = Field(default_factory=list, description="Concrete outputs to verify completion")
    supporting_resources: List[SupportingResource] = Field(
        default_factory=list, description="Grounded research papers, datasets, code repos"
    )


class ResearchRoadmap(BaseModel):
    """Complete 7-stage engineering implementation roadmap."""

    roadmap_id: str = Field(..., description="Unique roadmap ID")
    title: str = Field(..., description="Roadmap title based on query and research pool")
    target_query: str = Field(..., description="Research query guiding implementation")
    stages: List[RoadmapMilestone] = Field(..., description="Ordered 7-stage milestones")
    supporting_papers: List[str] = Field(default_factory=list, description="IDs of papers informing the roadmap")
    total_estimated_duration: str = Field(..., description="Aggregate estimated timeline")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp",
    )


# =========================================================================
# 5. AGGREGATED INTELLIGENCE REPORT
# =========================================================================

class ResearchIntelligenceReport(BaseModel):
    """Full intelligence report packaging all Phase 8 deliverables."""

    query: str = Field(..., description="Query or topic")
    run_id: Optional[str] = Field(default=None, description="Associated ranking run ID")
    explanations: List[RankingExplanation] = Field(default_factory=list, description="Itemized explanations")
    buildability_reports: List[BuildabilityReport] = Field(default_factory=list, description="Buildability evaluations")
    knowledge_graph: KnowledgeGraph = Field(..., description="Extracted scholarly knowledge graph")
    roadmap: ResearchRoadmap = Field(..., description="Actionable engineering roadmap")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp",
    )
