"""
Research Intelligence package (Phase 8):
- Explainability
- Buildability
- Knowledge Graph
- Engineering Roadmap
"""

from app.intelligence.buildability import BuildabilityEvaluator
from app.intelligence.explainer import RankingExplainer
from app.intelligence.graph import KnowledgeGraphBuilder
from app.intelligence.models import (
    BuildabilityDimension,
    BuildabilityReport,
    ComponentContribution,
    EdgeType,
    GraphEdge,
    GraphNode,
    GraphProvenance,
    KnowledgeGraph,
    NodeType,
    RankingExplanation,
    ResearchIntelligenceReport,
    ResearchRoadmap,
    RoadmapMilestone,
    RoadmapStageType,
    SupportingResource,
)
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
    "ResearchRoadmap",
    "ResearchIntelligenceReport",
    # Engines & Services
    "RankingExplainer",
    "BuildabilityEvaluator",
    "KnowledgeGraphBuilder",
    "RoadmapGenerator",
    "ResearchIntelligenceService",
]
