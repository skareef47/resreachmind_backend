"""
User preferences API endpoints for ResearchMind AI (Phase 9).

Connects frontend preferences configuration:
- GET /api/v1/preferences: Get user preferences
- PUT /api/v1/preferences: Update user preferences
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repository import DatabaseRepository
from app.db.session import get_db_session
from app.preferences.models import UserPreferences

logger = logging.getLogger("researchmind.api.preferences")

router = APIRouter(prefix="/preferences", tags=["Preferences"])


class UpdatePreferencesRequest(BaseModel):
    user_id: str = Field(default="default_user", description="User identifier")
    preferences: UserPreferences = Field(..., description="Updated preference configuration")


@router.get(
    "",
    response_model=UserPreferences,
    summary="Get User Preferences",
    description="Retrieve multi-dimensional preference profile for a user.",
)
async def get_preferences(
    user_id: str = Query(default="default_user", description="User identifier"),
    session: AsyncSession = Depends(get_db_session),
) -> UserPreferences:
    repo = DatabaseRepository(session)
    prefs = await repo.get_preferences(user_id=user_id)
    if not prefs:
        # Default baseline preferences
        return UserPreferences()
    return prefs


@router.put(
    "",
    response_model=UserPreferences,
    summary="Update User Preferences",
    description="Save or update multi-dimensional preference profile for a user.",
)
async def update_preferences(
    request: UpdatePreferencesRequest,
    session: AsyncSession = Depends(get_db_session),
) -> UserPreferences:
    repo = DatabaseRepository(session)
    await repo.save_preferences(request.preferences, user_id=request.user_id)
    return request.preferences
