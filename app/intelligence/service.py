"""
Research Intelligence Service orchestrator for ResearchMind AI.

Coordinates Explainability, Buildability Assessment, Knowledge Graph Construction,
and Actionable Roadmap Generation.
"""

import logging
from typing import List, Optional

from app.intelligence.buildability import BuildabilityEvaluator
from app.intelligence.explainer import RankingExplainer
from app.intelligence.graph import KnowledgeGraphBuilder
from app.intelligence.models import (
    BuildabilityReport,
    KnowledgeGraph,
    RankingExplanation,
    ResearchIntelligenceReport,
    ResearchRoadmap,
)
from app.intelligence.roadmap import RoadmapGenerator
from app.optimization.utility import UtilityRunStore, get_utility_run_store
from app.schemas.paper import ResearchDocument

logger = logging.getLogger("researchmind.intelligence.service")


class ResearchIntelligenceService:
    """
    Central service for Research Intelligence (Phase 8).
    """

    def __init__(self, run_store: Optional[UtilityRunStore] = None) -> None:
        self.run_store = run_store or get_utility_run_store()
        self.explainer = RankingExplainer(run_store=self.run_store)
        self.buildability_evaluator = BuildabilityEvaluator()
        self.graph_builder = KnowledgeGraphBuilder()
        self.roadmap_generator = RoadmapGenerator()

    def explain_run(self, run_id: str, document_id: Optional[str] = None) -> List[RankingExplanation]:
        """Explain ranking decisions for a stored run or a specific document in it."""
        if document_id:
            run_record = self.run_store.get_run(run_id)
            if run_record is None:
                raise KeyError(f"Run ID '{run_id}' not found in audit store")
            return [self.explainer.explain_candidate(run_record, document_id)]
        return self.explainer.explain_run(run_id)

    def evaluate_buildability(self, documents: List[ResearchDocument]) -> List[BuildabilityReport]:
        """Evaluate 9-dimensional buildability for a list of papers."""
        return self.buildability_evaluator.evaluate_documents(documents)

    def build_knowledge_graph(self, documents: List[ResearchDocument]) -> KnowledgeGraph:
        """Construct multi-entity knowledge graph with provenance."""
        return self.graph_builder.build_graph(documents)

    def generate_roadmap(self, query: str, documents: List[ResearchDocument]) -> ResearchRoadmap:
        """Generate 7-stage engineering roadmap grounded in research."""
        return self.roadmap_generator.generate_roadmap(query=query, documents=documents)

    def generate_full_report(
        self,
        query: str,
        documents: List[ResearchDocument],
        run_id: Optional[str] = None,
    ) -> ResearchIntelligenceReport:
        """Generate comprehensive Research Intelligence Report combining all dimensions."""
        explanations: List[RankingExplanation] = []
        if run_id:
            try:
                explanations = self.explain_run(run_id)
            except Exception as exc:
                logger.warning("Could not fetch explanations for run '%s': %s", run_id, exc)

        buildability_reports = self.evaluate_buildability(documents)
        kg = self.build_knowledge_graph(documents)
        roadmap = self.generate_roadmap(query=query, documents=documents)

        return ResearchIntelligenceReport(
            query=query,
            run_id=run_id,
            explanations=explanations,
            buildability_reports=buildability_reports,
            knowledge_graph=kg,
            roadmap=roadmap,
        )
