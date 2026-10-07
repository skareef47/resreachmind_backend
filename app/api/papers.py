"""
Paper details and bookmarking API endpoints for ResearchMind AI (Phase 9).

Connects frontend functionality:
- GET /api/v1/papers/{internal_id}: Full document details
- GET /api/v1/papers/saved: Saved papers library
- POST /api/v1/papers/saved: Save a paper
- DELETE /api/v1/papers/saved/{document_id}: Remove a saved paper
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repository import DatabaseRepository
from app.db.session import get_db_session
from app.schemas.paper import ResearchDocument

logger = logging.getLogger("researchmind.api.papers")

router = APIRouter(prefix="/papers", tags=["Papers"])


class SavePaperRequest(BaseModel):
    document_id: str = Field(..., description="Internal document identifier to save")
    user_id: str = Field(default="default_user", description="User identifier")
    notes: Optional[str] = Field(default=None, description="Personal research notes")
    tags: Optional[List[str]] = Field(default=None, description="Custom organization tags")


class SavedPaperResponse(BaseModel):
    id: str
    document_id: str
    user_id: str
    notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    saved_at: str
    document: Optional[ResearchDocument] = None


@router.get(
    "/saved",
    response_model=List[SavedPaperResponse],
    summary="Get Saved Papers",
    description="Retrieve all research papers bookmarked by the user.",
)
async def get_saved_papers(
    user_id: str = "default_user",
    session: AsyncSession = Depends(get_db_session),
) -> List[SavedPaperResponse]:
    repo = DatabaseRepository(session)
    entries = await repo.get_saved_papers(user_id=user_id)
    results = []
    for e in entries:
        doc_schema = repo._to_schema_document(e.document) if e.document else None
        results.append(
            SavedPaperResponse(
                id=e.id,
                document_id=e.document_id,
                user_id=e.user_id,
                notes=e.notes,
                tags=e.tags or [],
                saved_at=e.saved_at.isoformat(),
                document=doc_schema,
            )
        )
    return results


@router.post(
    "/saved",
    response_model=SavedPaperResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save / Bookmark Paper",
    description="Save a research paper to the user's library with optional notes and tags.",
)
async def save_paper(
    request: SavePaperRequest,
    session: AsyncSession = Depends(get_db_session),
) -> SavedPaperResponse:
    repo = DatabaseRepository(session)
    # Check document exists
    doc = await repo.get_document(request.document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{request.document_id}' not found in canonical database.",
        )
    saved = await repo.save_paper(
        document_id=request.document_id,
        user_id=request.user_id,
        notes=request.notes,
        tags=request.tags,
    )
    return SavedPaperResponse(
        id=saved.id,
        document_id=saved.document_id,
        user_id=saved.user_id,
        notes=saved.notes,
        tags=saved.tags or [],
        saved_at=saved.saved_at.isoformat(),
        document=doc,
    )


@router.delete(
    "/saved/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Remove Saved Paper",
    description="Remove a research paper from the user's library.",
)
async def remove_saved_paper(
    document_id: str,
    user_id: str = "default_user",
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    repo = DatabaseRepository(session)
    deleted = await repo.remove_saved_paper(document_id=document_id, user_id=user_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Saved paper '{document_id}' not found for user '{user_id}'.",
        )
    return {"deleted": True, "status": "removed", "document_id": document_id}


@router.get(
    "/{internal_id}",
    response_model=ResearchDocument,
    summary="Get Paper Details",
    description="Retrieve full canonical metadata, citations, and provenance for a paper.",
)
async def get_paper_details(
    internal_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> ResearchDocument:
    repo = DatabaseRepository(session)
    doc = await repo.get_document(internal_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research paper '{internal_id}' not found.",
        )
    return doc
