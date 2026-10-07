"""
Database session and connection management for ResearchMind AI (Phase 9).

Supports asynchronous connection to PostgreSQL and SQLite (fallback) via SQLAlchemy.
"""

import logging
import time
from typing import AsyncGenerator, Dict, Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings
from app.db.models import Base

logger = logging.getLogger("researchmind.db.session")

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Retrieve or initialize singleton async database engine."""
    global _engine, _sessionmaker
    if _engine is None:
        settings = get_settings()
        db_url = settings.DATABASE_URL

        # Special configuration for SQLite
        connect_args = {}
        if "sqlite" in db_url:
            connect_args = {"check_same_thread": False}

        _engine = create_async_engine(
            db_url,
            echo=settings.DATABASE_ECHO,
            future=True,
            connect_args=connect_args,
        )
        _sessionmaker = async_sessionmaker(
            bind=_engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
        logger.info("Initialized database engine with URL dialect: %s", _engine.dialect.name)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Retrieve async sessionmaker."""
    global _sessionmaker
    if _sessionmaker is None:
        get_engine()
    return _sessionmaker  # type: ignore


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding an async database session."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema initialized successfully.")


async def check_db_health() -> Dict[str, Any]:
    """Ping database and measure response latency."""
    engine = get_engine()
    start_time = time.perf_counter()
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        return {
            "status": "connected",
            "dialect": engine.dialect.name,
            "latency_ms": latency_ms,
        }
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        logger.warning("Database health check failed: %s", exc)
        return {
            "status": "degraded",
            "error": str(exc),
            "latency_ms": latency_ms,
        }
