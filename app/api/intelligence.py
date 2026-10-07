"""
Research Intelligence API routes (Phase 8).

Provides versioned endpoints under /api/v1/intelligence for:
- Explainability (/explain)
- Buildability Evaluation (/buildability)
- Knowledge Graph Synthesis (/graph)
- Engineering Roadmap Generation (/roadmap)
- Comprehensive Research Intelligence Report (/report)
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.intelligence.models import (
    BuildabilityReport,
    KnowledgeGraph,
    RankingExplanation,
    ResearchIntelligenceReport,
    ResearchRoadmap,
)
from app.intelligence.service import ResearchIntelligenceService
from app.schemas.paper import ResearchDocument

logger = logging.getLogger("researchmind.api.intelligence")

router = APIRouter(prefix="/intelligence", tags=["Intelligence"])

_service = ResearchIntelligenceService()


# -------------------------------------------------------------------------
# Request Models
# -------------------------------------------------------------------------

class ExplainRequest(BaseModel):
    run_id: str = Field(..., description="Immutable ranking run ID to explain")
    document_id: Optional[str] = Field(default=None, description="Optional specific document ID to explain")


class BuildabilityRequest(BaseModel):
    documents: List[ResearchDocument] = Field(..., description="Papers to evaluate for buildability")


class KnowledgeGraphRequest(BaseModel):
    documents: List[ResearchDocument] = Field(..., description="Papers to synthesize into a knowledge graph")


class RoadmapRequest(BaseModel):
    query: str = Field(..., description="Research query or topic guiding implementation")
    documents: List[ResearchDocument] = Field(..., description="Supporting papers informing the roadmap")


class IntelligenceReportRequest(BaseModel):
    query: str = Field(..., description="Research query")
    documents: List[ResearchDocument] = Field(..., description="Retrieved/ranked documents")
    run_id: Optional[str] = Field(default=None, description="Optional ranking run ID for explanations")


# -------------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------------

@router.get(
    "/explain/{run_id}",
    response_model=List[RankingExplanation],
    summary="Explain Ranking Decisions (GET)",
    description="Retrieve transparent ranking explanations derived strictly from stored run audit telemetry.",
)
def get_run_explanations(
    run_id: str,
    document_id: Optional[str] = Query(default=None, description="Filter for specific document ID"),
) -> List[RankingExplanation]:
    try:
        return _service.explain_run(run_id=run_id, document_id=document_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.exception("Error retrieving explanation: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post(
    "/explain",
    response_model=List[RankingExplanation],
    summary="Explain Ranking Decisions (POST)",
    description="Retrieve transparent explanations for a run or document via POST payload.",
)
def explain_ranking(request: ExplainRequest) -> List[RankingExplanation]:
    try:
        return _service.explain_run(run_id=request.run_id, document_id=request.document_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.exception("Error in explain_ranking: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post(
    "/buildability",
    response_model=List[BuildabilityReport],
    summary="Evaluate Engineering Buildability",
    description="Assess 9 dimensions of engineering buildability and reproducibility with factual evidence.",
)
def evaluate_buildability(request: BuildabilityRequest) -> List[BuildabilityReport]:
    try:
        return _service.evaluate_buildability(request.documents)
    except Exception as exc:
        logger.exception("Error evaluating buildability: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post(
    "/graph",
    response_model=KnowledgeGraph,
    summary="Construct Scholarly Knowledge Graph",
    description="Synthesize multi-entity relationships (Paper -> Dataset/Model/Method/Repo/Related Paper) with provenance.",
)
def construct_knowledge_graph(request: KnowledgeGraphRequest) -> KnowledgeGraph:
    try:
        return _service.build_knowledge_graph(request.documents)
    except Exception as exc:
        logger.exception("Error constructing knowledge graph: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post(
    "/roadmap",
    response_model=ResearchRoadmap,
    summary="Generate 7-Stage Engineering Roadmap",
    description="Generate a 7-stage actionable execution roadmap linked to supporting research papers, datasets, and code.",
)
def generate_roadmap(request: RoadmapRequest) -> ResearchRoadmap:
    try:
        return _service.generate_roadmap(query=request.query, documents=request.documents)
    except Exception as exc:
        logger.exception("Error generating roadmap: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post(
    "/report",
    response_model=ResearchIntelligenceReport,
    summary="Generate Comprehensive Research Intelligence Report",
    description="Aggregate Explainability, Buildability Assessment, Knowledge Graph, and Roadmap into a unified report.",
)
def generate_intelligence_report(request: IntelligenceReportRequest) -> ResearchIntelligenceReport:
    try:
        return _service.generate_full_report(
            query=request.query,
            documents=request.documents,
            run_id=request.run_id,
        )
    except Exception as exc:
        logger.exception("Error generating intelligence report: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
