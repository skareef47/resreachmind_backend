"""
Research Intelligence package (Phase 8):
- Explainability
- Buildability
- Knowledge Graph
- Engineering Roadmap
"""

from app.intelligence.architecture import ArchitectureGenerator
from app.intelligence.buildability import BuildabilityEvaluator
from app.intelligence.explainer import RankingExplainer
from app.intelligence.graph import KnowledgeGraphBuilder
from app.intelligence.models import (
    ArchitectureComponent,
    ArchitectureConnection,
    BuildabilityDimension,
    BuildabilityReport,
    CompatibilityAnalysis,
    ComponentContribution,
    DependencyItem,
    EdgeType,
    EvidenceStatus,
    GraphEdge,
    GraphNode,
    GraphProvenance,
    HardwareRequirement,
    KnowledgeGraph,
    NodeType,
    RankingExplanation,
    ResearchArchitectureResponse,
    ResearchIntelligenceReport,
    ResearchRoadmap,
    ResearchRoadmapResponse,
    RoadmapDependencyEdge,
    RoadmapMilestone,
    RoadmapPhase,
    RoadmapStageType,
    SupportingResource,
    TechnicalArchitecture,
    GapType,
    SupportingPaperRef,
    PotentialResearchGap,
    ResearchGapLandscapeSummary,
    ResearchGapsResponse,
)
from app.intelligence.gaps import ResearchGapAnalyzer
from app.intelligence.paper_intelligence import PaperIntelligenceAnalyzer
from app.intelligence.roadmap import RoadmapGenerator
from app.intelligence.service import ResearchIntelligenceService

__all__ = [
    # Models
    "ComponentContribution",
    "RankingExplanation",
    "BuildabilityDimension",
    "BuildabilityReport",
    "NodeType",
    "EdgeType",
    "GraphProvenance",
    "GraphNode",
    "GraphEdge",
    "KnowledgeGraph",
    "RoadmapStageType",
    "SupportingResource",
    "RoadmapMilestone",
    "RoadmapPhase",
    "RoadmapDependencyEdge",
    "ResearchRoadmap",
    "ResearchRoadmapResponse",
    "ResearchIntelligenceReport",
    "EvidenceStatus",
    "ArchitectureComponent",
    "ArchitectureConnection",
    "HardwareRequirement",
    "DependencyItem",
    "CompatibilityAnalysis",
    "TechnicalArchitecture",
    "ResearchArchitectureResponse",
    "GapType",
    "SupportingPaperRef",
    "PotentialResearchGap",
    "ResearchGapLandscapeSummary",
    "ResearchGapsResponse",
    # Engines & Services
    "RankingExplainer",
    "BuildabilityEvaluator",
    "KnowledgeGraphBuilder",
    "RoadmapGenerator",
    "ArchitectureGenerator",
    "ResearchGapAnalyzer",
    "PaperIntelligenceAnalyzer",
    "ResearchIntelligenceService",
]
