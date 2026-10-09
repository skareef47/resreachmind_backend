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
from pydantic import BaseModel, Field, model_validator, field_serializer


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
    status: str = Field(default="unknown", description="Evidence status: 'verified', 'unknown', or 'absent'")
    evidence: List[str] = Field(default_factory=list, description="Concrete factual evidence items")
    evidence_summary: str = Field(default="", description="Summary explanation or 'Not verified from available sources'")
    source: Optional[str] = Field(default=None, description="Identified source or provider")
    source_url: Optional[str] = Field(default=None, description="External URL for verifiable evidence")
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
    feature_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Feature breakdown of core buildability dimensions (dataset, code, model, docs, reproducibility)",
    )
    evidence_breakdown: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Detailed evidence provenance breakdown by dimension (dataset, code, etc.)",
    )
    evidence_sources: List[str] = Field(
        default_factory=list,
        description="Verifiable URLs and source links supporting the buildability assessment",
    )
    unknown_fields: List[str] = Field(
        default_factory=list,
        description="Dimensions where evidence could not be verified from available sources",
    )
    evaluated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of evaluation",
    )


# =========================================================================
# 3. KNOWLEDGE GRAPH MODELS
# =========================================================================

class NodeType(str, Enum):
    PAPER = "paper"
    AUTHOR = "author"
    INSTITUTION = "institution"
    METHOD = "method"
    MODEL = "model"
    DATASET = "dataset"
    TASK = "task"
    VENUE = "venue"
    CONCEPT = "concept"
    TECHNOLOGY = "technology"
    REPOSITORY = "repository"


class EdgeType(str, Enum):
    USES = "uses"
    USES_DATASET = "uses_dataset"
    PRODUCES_DATASET = "produces_dataset"
    USES_MODEL = "uses_model"
    PROPOSES_MODEL = "proposes_model"
    USES_METHOD = "uses_method"
    PROPOSES_METHOD = "proposes_method"
    WRITTEN_BY = "written_by"
    AFFILIATED_WITH = "affiliated_with"
    PUBLISHED_IN = "published_in"
    HAS_CONCEPT = "has_concept"
    ADDRESSES = "addresses"
    ADDRESSES_TASK = "addresses_task"
    USES_TECHNOLOGY = "uses_technology"
    HAS_REPOSITORY = "has_repository"
    CITES_PAPER = "cites_paper"
    SIMILAR_TO_PAPER = "similar_to_paper"
    RELATED_TO = "related_to"


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
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Entity attributes / metadata")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Entity attributes alias for compatibility")
    provenance: Optional[GraphProvenance] = Field(default=None, description="Data provenance")


class GraphEdge(BaseModel):
    """Directed semantic relationship edge in the knowledge graph."""

    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    type: EdgeType = Field(..., description="Relationship semantic category")
    label: str = Field(default="", description="Relationship display label")
    evidence: str = Field(default="", description="Concrete textual evidence or rationale supporting relationship")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Relationship extraction confidence")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relationship metadata")
    provenance: Optional[GraphProvenance] = Field(default=None, description="Relationship provenance")


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


class RoadmapTask(BaseModel):
    """Actionable implementation task directly derived from architecture and research evidence."""

    phase: int = Field(..., description="Phase index (1 to 8)")
    task: str = Field(..., description="Name or title of the implementation task")
    description: str = Field(..., description="Concrete engineering description")
    dependency: str = Field(default="None", description="Direct prerequisite task or phase")
    input: str = Field(default="", description="Input artifacts or prerequisites")
    output: str = Field(default="", description="Produced deliverable or artifact")
    technology: str = Field(default="", description="Primary technology or framework used")
    difficulty: str = Field(default="Medium", description="Estimated task difficulty: Low, Medium, High, Critical")
    evidence_status: str = Field(default="VERIFIED", description="VERIFIED, INFERRED, or UNKNOWN")
    evidence_source: Optional[str] = Field(default=None, description="Paper ID / section / evidence citation")
    supporting_papers: List[str] = Field(default_factory=list, description="Paper titles or IDs supporting this task")


class RoadmapPhase(BaseModel):
    """
    Actionable implementation roadmap phase with tasks, dependencies,
    inputs, outputs, and attached evidence.
    """

    phase: int = Field(..., description="Phase index (1 to 8)")
    title: str = Field(..., description="Phase title")
    description: str = Field(..., description="Actionable execution description")
    tasks: List[str] = Field(default_factory=list, description="Concrete technical task titles")
    task_items: List[RoadmapTask] = Field(default_factory=list, description="Paper-specific structured roadmap tasks")
    dependencies: List[str] = Field(default_factory=list, description="Prerequisite phase names or IDs")
    inputs: List[str] = Field(default_factory=list, description="Inputs required to execute this phase")
    outputs: List[str] = Field(default_factory=list, description="Outputs and deliverables produced")
    estimated_difficulty: str = Field(default="Medium", description="Estimated difficulty: Low, Medium, High, Very High")
    evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Paper evidence grounding this phase")
    status: str = Field(default="planned", description="Status: planned, in_progress, completed, blocked")


class RoadmapDependencyEdge(BaseModel):
    """Directed transition between roadmap phases."""

    from_phase: int = Field(..., description="Preceding phase number")
    from_title: str = Field(..., description="Preceding phase title")
    to_phase: int = Field(..., description="Dependent phase number")
    to_title: str = Field(..., description="Dependent phase title")
    reason: str = Field(..., description="Why this dependency exists")


class ResearchRoadmap(BaseModel):
    """Complete engineering implementation roadmap synthesized from research."""

    roadmap_id: str = Field(..., description="Unique roadmap ID")
    title: str = Field(..., description="Roadmap title based on query and research pool")
    target_query: str = Field(default="", description="Research query guiding implementation")
    phases: List[RoadmapPhase] = Field(default_factory=list, description="Ordered 8 implementation phases")
    stages: List[RoadmapMilestone] = Field(default_factory=list, description="Ordered milestone stages")
    selected_paper_ids: List[str] = Field(default_factory=list, description="IDs of source papers used")
    papers_used: List[str] = Field(default_factory=list, description="IDs of source papers used")
    supporting_papers: List[str] = Field(default_factory=list, description="IDs of papers informing the roadmap")
    dependency_graph: List[RoadmapDependencyEdge] = Field(default_factory=list, description="Phase dependency flow")
    total_estimated_duration: str = Field(..., description="Aggregate estimated timeline")
    constraints_applied: Dict[str, Any] = Field(default_factory=dict, description="Active user constraints")
    evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Aggregated evidence log")
    architecture_hash: str = Field(default="", description="Deterministic SHA256 architecture fingerprint")
    query: Optional[str] = Field(default=None, description="Research query guiding implementation")
    summary: str = Field(default="", description="High-level engineering execution summary")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp",
    )

    @model_validator(mode="before")
    @classmethod
    def sync_paper_id_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("papers_used") and data.get("selected_paper_ids"):
                data["papers_used"] = data["selected_paper_ids"]
            elif not data.get("selected_paper_ids") and data.get("papers_used"):
                data["selected_paper_ids"] = data["papers_used"]
            if data.get("query") and not data.get("target_query"):
                data["target_query"] = data["query"]
            elif data.get("target_query") and not data.get("query"):
                data["query"] = data["target_query"]
        return data


# Alias for explicit API response type
ResearchRoadmapResponse = ResearchRoadmap



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


# =========================================================================
# 6. RESEARCH ARCHITECTURE GENERATION MODELS
# =========================================================================

class EvidenceStatus(str, Enum):
    """Grounding status of an architectural specification."""

    VERIFIED = "VERIFIED"
    INFERRED = "INFERRED"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            val_upper = value.upper()
            for member in cls:
                if member.name == val_upper or member.value == val_upper:
                    return member
        return None


class ArchitectureComponent(BaseModel):
    """Concrete technical module grounded in paper evidence."""

    component: str = Field(..., description="Architecture stage / component category")
    value: str = Field(..., description="Specified technology, dataset, model, or implementation method")
    source_paper: str = Field(..., description="Title or identifier of the grounding research paper")
    evidence: str = Field(..., description="Verbatim text quote, provenance field, or derivation basis")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction or inference confidence")
    status: EvidenceStatus = Field(default=EvidenceStatus.VERIFIED, description="Grounding status")
    evidence_status: EvidenceStatus = Field(default=EvidenceStatus.VERIFIED, description="Explicit uppercase grounding status")
    evidence_source: str = Field(default="", description="Paper ID, section, or extracted evidence provenance")
    technologies: List[str] = Field(default_factory=list, description="Technologies or libraries utilized")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters or configuration")
    details: Dict[str, Any] = Field(default_factory=dict, description="Component parameters or metadata")
    supporting_paper_id: Optional[str] = Field(default=None, description="Internal ID of supporting paper")

    @field_serializer("status")
    def serialize_status(self, v: Any) -> str:
        if hasattr(v, "value"):
            return str(v.value).lower()
        return str(v).lower()

    @field_serializer("evidence_status")
    def serialize_evidence_status(self, v: Any) -> str:
        if hasattr(v, "value"):
            return str(v.value).upper()
        return str(v).upper()

    @model_validator(mode="before")
    @classmethod
    def sync_status_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            st = data.get("evidence_status") or data.get("status")
            if st:
                data["status"] = st
                data["evidence_status"] = st
            if not data.get("evidence_source"):
                data["evidence_source"] = data.get("source_paper", "")
        return data


class ArchitectureConnection(BaseModel):
    """Directed interface between architecture components."""

    source: str = Field(..., description="Origin component")
    target: str = Field(..., description="Destination component")
    description: str = Field(..., description="Nature of the integration")
    data_flow: str = Field(..., description="Payload or tensor format passed along this edge")


class HardwareRequirement(BaseModel):
    """Compute, accelerator, or memory requirement for the pipeline."""

    category: str = Field(..., description="Hardware type (e.g. GPU, VRAM, RAM, CPU, Storage)")
    requirement: str = Field(..., description="Specific specification (e.g. '>= 16GB VRAM')")
    status: EvidenceStatus = Field(default=EvidenceStatus.VERIFIED, description="Grounding status")
    evidence: str = Field(..., description="Direct paper evidence or benchmark profile")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence")

    @field_serializer("status")
    def serialize_status(self, v: Any) -> str:
        if hasattr(v, "value"):
            return str(v.value).lower()
        return str(v).lower()


class DependencyItem(BaseModel):
    """Software package or environment dependency."""

    name: str = Field(..., description="Package or runtime name (e.g. 'torch', 'transformers')")
    version_spec: Optional[str] = Field(default=None, description="Required version or constraint")
    category: str = Field(default="runtime", description="Category: framework, runtime, api, data, driver")
    license: Optional[str] = Field(default=None, description="Package license")
    status: EvidenceStatus = Field(default=EvidenceStatus.VERIFIED, description="Grounding status")
    evidence: str = Field(..., description="Evidence quote or repository configuration")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence")

    @field_serializer("status")
    def serialize_status(self, v: Any) -> str:
        if hasattr(v, "value"):
            return str(v.value).lower()
        return str(v).lower()


class CompatibilityAnalysis(BaseModel):
    """Synergy and conflict assessment across multiple selected papers."""

    is_compatible: bool = Field(default=True, description="True if no blocking conflicts exist")
    compatibility_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Aggregate synergy score")
    compatible_points: List[str] = Field(default_factory=list, description="Shared stacks, complementary roles")
    conflicts: List[str] = Field(default_factory=list, description="License, framework, or hardware clashes")


class TechnicalArchitecture(BaseModel):
    """Concrete technical implementation architecture."""

    components: List[ArchitectureComponent] = Field(..., description="Ordered system modules")
    connections: List[ArchitectureConnection] = Field(..., description="Data flow interconnections")
    hardware: List[HardwareRequirement] = Field(..., description="Compute environment requirements")
    dependencies: List[DependencyItem] = Field(..., description="Software stack dependencies")


class ResearchArchitectureResponse(BaseModel):
    """Top-level response payload for POST /api/v1/architecture."""

    architecture_id: str = Field(..., description="Unique architecture specification ID")
    title: str = Field(..., description="Title describing the synthesized system architecture")
    query: Optional[str] = Field(default=None, description="Research query guiding architecture")
    selected_paper_ids: List[str] = Field(default_factory=list, description="IDs of source papers used")
    papers_used: List[str] = Field(default_factory=list, description="IDs/titles of source papers used")
    architecture_hash: str = Field(default="", description="Deterministic SHA256 architecture fingerprint")
    components: List[ArchitectureComponent] = Field(default_factory=list, description="Direct list of architecture components")
    connections: List[ArchitectureConnection] = Field(default_factory=list, description="Direct list of architecture connections")
    architecture: TechnicalArchitecture = Field(..., description="Core architecture graph and specs")
    evidence: List[ArchitectureComponent] = Field(default_factory=list, description="Complete evidence log")
    architecture_type: str = Field(default="Custom Research Architecture", description="Identified architectural topology")
    confidence: float = Field(default=0.88, ge=0.0, le=1.0, description="Overall architectural confidence")
    is_generic_scaffold: bool = Field(default=False, description="Flag indicating if evidence was insufficient")
    compatibility: CompatibilityAnalysis = Field(
        default_factory=CompatibilityAnalysis, description="Multi-paper compatibility analysis"
    )
    constraints_applied: Dict[str, Any] = Field(default_factory=dict, description="User constraints integrated")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp",
    )

    @model_validator(mode="before")
    @classmethod
    def populate_root_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            arch = data.get("architecture")
            if arch:
                if isinstance(arch, dict):
                    if not data.get("components") and "components" in arch:
                        data["components"] = arch["components"]
                    if not data.get("connections") and "connections" in arch:
                        data["connections"] = arch["connections"]
                elif hasattr(arch, "components"):
                    if not data.get("components"):
                        data["components"] = list(arch.components)
                    if not data.get("connections") and hasattr(arch, "connections"):
                        data["connections"] = list(arch.connections)
            if not data.get("papers_used") and data.get("selected_paper_ids"):
                data["papers_used"] = data["selected_paper_ids"]
            elif not data.get("selected_paper_ids") and data.get("papers_used"):
                data["selected_paper_ids"] = data["papers_used"]
        return data


# =========================================================================
# 6. RESEARCH GAP INTELLIGENCE MODELS
# =========================================================================

class GapType(str, Enum):
    METHOD = "method"
    DATASET = "dataset"
    EVALUATION = "evaluation"
    APPLICATION = "application"
    SCALABILITY = "scalability"


class SupportingPaperRef(BaseModel):
    """Reference to an actual candidate paper supporting a gap observation."""

    id: str = Field(..., description="Document internal identifier")
    title: str = Field(..., description="Paper title")
    citation: Optional[str] = Field(default=None, description="Author and year citation, venue, or DOI")
    relevance_to_gap: Optional[str] = Field(default=None, description="How this paper supports the gap observation")


class PotentialResearchGap(BaseModel):
    """
    Evidence-backed potential research gap identified across retrieved research papers.
    Strictly qualifies observations: does not claim global novelty across all literature.
    """

    id: str = Field(..., description="Unique gap identifier (e.g. 'gap_eval_robustness_01')")
    title: str = Field(..., description="Potential research gap title (e.g. 'Potential Research Gap: ...')")
    description: str = Field(..., description="Clear explanation of the underexplored area")
    gap_type: GapType = Field(..., description="Category: method, dataset, evaluation, application, scalability")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Evidence confidence score [0.0, 1.0]")
    evidence: List[str] = Field(default_factory=list, description="Concrete textual or statistical evidence items")
    supporting_papers: List[SupportingPaperRef] = Field(
        default_factory=list, description="Actual retrieved research papers supporting this gap observation"
    )
    why_it_appears: str = Field(
        ..., description="Rationale for why this appears as an underexplored area in the candidate set"
    )
    observation: str = Field(
        ..., description="Empirical, evidence-backed observation grounded strictly in the retrieved corpus"
    )
    hypothesis: str = Field(
        ..., description="AI-generated research hypothesis articulating the unverified proposition"
    )
    possible_research_direction: str = Field(
        ..., description="Concrete research opportunity, proposed methodology, or experimental study design"
    )
    coverage_context: Dict[str, Any] = Field(
        default_factory=dict, description="Quantitative candidate set distribution (e.g. prevalence vs absence)"
    )


class ResearchGapLandscapeSummary(BaseModel):
    """Summary of methodology and dataset distributions across the candidate set."""

    total_papers_analyzed: int = Field(..., description="Total candidate papers evaluated")
    dominant_methods: List[str] = Field(default_factory=list, description="High-frequency methodologies")
    dominant_datasets: List[str] = Field(default_factory=list, description="High-frequency benchmark datasets")
    dominant_models: List[str] = Field(default_factory=list, description="High-frequency model architectures")
    underrepresented_areas: List[str] = Field(default_factory=list, description="Identified low-coverage domains")
    frequently_cited_limitations: List[str] = Field(default_factory=list, description="Common limitation patterns")
    coverage_analysis_scope: str = Field(
        default="Limited evidence was found within the analyzed candidate set.",
        description="Standardized scope qualification",
    )


class ResearchGapsResponse(BaseModel):
    """Top-level response payload for POST /api/v1/research-gaps."""

    query: Optional[str] = Field(default=None, description="Search or synthesis query")
    candidate_papers_analyzed: int = Field(..., description="Number of papers included in coverage analysis")
    potential_gaps: List[PotentialResearchGap] = Field(default_factory=list, description="Identified potential gaps")
    landscape_summary: ResearchGapLandscapeSummary = Field(..., description="Research landscape analysis summary")
    disclaimer: str = Field(
        default="Potential research gaps represent underexplored areas observed within the analyzed candidate set. Limited evidence was found within the analyzed candidate set, which does not constitute proof that no prior work exists across all scholarly literature.",
        description="Scientific scope disclaimer",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp",
    )


