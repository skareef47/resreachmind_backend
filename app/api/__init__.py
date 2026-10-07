"""API routes aggregator."""

from fastapi import APIRouter

from app.api.evaluation import router as evaluation_router
from app.api.feedback import router as feedback_router
from app.api.health import router as health_router
from app.api.intelligence import router as intelligence_router
from app.api.observability import router as observability_router
from app.api.optimization import router as optimization_router
from app.api.papers import router as papers_router
from app.api.preferences import router as preferences_router
from app.api.search import router as search_router
from app.api.utility import router as utility_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(search_router)
api_router.include_router(papers_router)
api_router.include_router(feedback_router)
api_router.include_router(preferences_router)
api_router.include_router(utility_router)
api_router.include_router(optimization_router)
api_router.include_router(intelligence_router)
api_router.include_router(observability_router)
api_router.include_router(evaluation_router)

__all__ = ["api_router"]

