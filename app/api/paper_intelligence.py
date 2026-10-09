"""
Paper Intelligence API endpoint for ResearchMind AI.

Provides versioned endpoint:
- POST /api/v1/paper-intelligence

Extracts comprehensive, evidence-grounded paper-specific intelligence:
1. Paper Summary & Research Problem
2. Verified/Inferred Datasets, Methodology, and Models
3. Key Empirical Results
4. Grounded Limitations (EXPLICIT, DERIVED, UNKNOWN)
5. How to Overcome (Author-Proposed vs ResearchMind-Proposed)
6. Research Gaps & Actionable Opportunities
7. Provenance Evidence Audit Trail
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repository import DatabaseRepository
from app.db.session import get_db_session
from app.intelligence.paper_intelligence import PaperIntelligenceAnalyzer
from app.schemas.paper import ResearchDocument
from app.schemas.paper_intelligence import (
    PaperIntelligenceRequest,
    PaperIntelligenceResponse,
)

logger = logging.getLogger("researchmind.api.paper_intelligence")

router = APIRouter(tags=["Paper Intelligence"])
analyzer = PaperIntelligenceAnalyzer()


@router.post(
    "/paper-intelligence",
    response_model=PaperIntelligenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Paper-Specific Intelligence",
    description="Analyze a single research paper to extract summary, limitations, mitigations, gaps, and opportunities.",
)
async def generate_paper_intelligence(
    payload: PaperIntelligenceRequest,
    session: AsyncSession = Depends(get_db_session),
) -> PaperIntelligenceResponse:
    """Analyze a single selected research paper based on its actual available evidence."""
    target_paper = payload.paper
    target_id = payload.paper_id

    # If full paper dictionary not provided in payload, attempt lookup from database
    if not target_paper and target_id:
        repo = DatabaseRepository(session)
        db_doc = await repo.get_document(target_id)
        if db_doc:
            target_paper = repo._to_schema_document(db_doc).model_dump()
        else:
            # Check by source_id
            db_doc_source = await repo.get_document_by_source_id(target_id)
            if db_doc_source:
                target_paper = repo._to_schema_document(db_doc_source).model_dump()

    if not target_paper and not target_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'paper' payload dictionary or a valid 'paper_id' must be provided.",
        )

    if not target_paper:
        # Fallback minimal stub when paper not found in local DB but ID was supplied
        target_paper = {
            "internal_id": target_id,
            "title": f"Paper ({target_id})",
            "abstract": "",
        }

    try:
        response = analyzer.analyze_paper(paper=target_paper, paper_id=target_id)
        return response
    except Exception as exc:
        logger.error("Failed to generate paper intelligence: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate paper intelligence: {str(exc)}",
        )
