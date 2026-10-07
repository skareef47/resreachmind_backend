"""
Async Database Repository for ResearchMind AI (Phase 9).

Provides clean data access patterns for:
- ResearchDocuments & Provenance
- Saved Papers
- User Feedback
- User Preferences
- Utility Runs
- Optimization Results
- Research Roadmaps
"""

import logging
from typing import List, Optional
from datetime import datetime, timezone
import uuid

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    OptimizationResultDB,
    ResearchDocumentDB,
    ResearchRoadmapDB,
    SavedPaperDB,
    UserFeedbackDB,
    UserPreferencesDB,
    UtilityRunDB,
)
from app.feedback.models import FeedbackItem, FeedbackType
from app.intelligence.models import ResearchRoadmap
from app.optimization.models import OptimizationResult, SolverStatus, SolverType, VerificationReport
from app.optimization.utility import (
    CandidateEvaluation,
    UtilityComponentScores,
    UtilityRunRecord,
    UtilityWeights,
)
from app.preferences.models import UserPreferences
from app.schemas.paper import ProvenanceRecord, ResearchDocument

logger = logging.getLogger("researchmind.db.repository")


class DatabaseRepository:
    """Async repository providing entity persistence and retrieval."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # -------------------------------------------------------------------------
    # 1. Research Documents
    # -------------------------------------------------------------------------
    async def upsert_document(self, doc: ResearchDocument) -> ResearchDocumentDB:
        existing = await self.session.get(ResearchDocumentDB, doc.internal_id)
        prov_list = [p.model_dump() for p in doc.provenance]

        if existing:
            existing.title = doc.title
            existing.abstract = doc.abstract
            existing.authors = doc.authors
            existing.publication_year = doc.publication_year
            existing.publication_date = doc.publication_date
            existing.doi = doc.doi
            existing.url = doc.url
            existing.citation_count = doc.citation_count
            existing.references = doc.references
            existing.citations = doc.citations
            existing.venue = doc.venue
            existing.fields_of_study = doc.fields_of_study
            existing.concepts = doc.concepts
            existing.datasets = doc.datasets
            existing.methods = doc.methods
            existing.models = doc.models
            existing.code_url = doc.code_url
            existing.open_access = doc.open_access
            existing.license = doc.license
            existing.provenance = prov_list
            existing.raw_source_metadata = doc.raw_source_metadata
            db_doc = existing
        else:
            db_doc = ResearchDocumentDB(
                internal_id=doc.internal_id,
                source=doc.source,
                source_id=doc.source_id,
                title=doc.title,
                abstract=doc.abstract,
                authors=doc.authors,
                publication_year=doc.publication_year,
                publication_date=doc.publication_date,
                doi=doc.doi,
                url=doc.url,
                citation_count=doc.citation_count or 0,
                references=doc.references,
                citations=doc.citations,
                venue=doc.venue,
                fields_of_study=doc.fields_of_study,
                concepts=doc.concepts,
                datasets=doc.datasets,
                methods=doc.methods,
                models=doc.models,
                code_url=doc.code_url,
                open_access=doc.open_access,
                license=doc.license,
                provenance=prov_list,
                raw_source_metadata=doc.raw_source_metadata,
                retrieval_timestamp=doc.retrieval_timestamp,
            )
            self.session.add(db_doc)

        await self.session.commit()
        await self.session.refresh(db_doc)
        return db_doc

    async def upsert_documents(self, docs: List[ResearchDocument]) -> List[ResearchDocumentDB]:
        results: List[ResearchDocumentDB] = []
        for d in docs:
            saved = await self.upsert_document(d)
            results.append(saved)
        return results

    async def get_document(self, internal_id: str) -> Optional[ResearchDocument]:
        db_doc = await self.session.get(ResearchDocumentDB, internal_id)
        if not db_doc:
            return None
        return self._to_schema_document(db_doc)

    async def list_documents(self, limit: int = 50, offset: int = 0) -> List[ResearchDocument]:
        stmt = select(ResearchDocumentDB).offset(offset).limit(limit)
        res = await self.session.execute(stmt)
        return [self._to_schema_document(r) for r in res.scalars().all()]

    def _to_schema_document(self, db_doc: ResearchDocumentDB) -> ResearchDocument:
        prov = [ProvenanceRecord(**p) for p in (db_doc.provenance or [])]
        return ResearchDocument(
            internal_id=db_doc.internal_id,
            source=db_doc.source,
            source_id=db_doc.source_id,
            title=db_doc.title,
            abstract=db_doc.abstract,
            authors=db_doc.authors or [],
            publication_year=db_doc.publication_year,
            publication_date=db_doc.publication_date,
            doi=db_doc.doi,
            url=db_doc.url,
            citation_count=db_doc.citation_count or 0,
            references=db_doc.references or [],
            citations=db_doc.citations or [],
            venue=db_doc.venue,
            fields_of_study=db_doc.fields_of_study or [],
            concepts=db_doc.concepts or [],
            datasets=db_doc.datasets or [],
            methods=db_doc.methods or [],
            models=db_doc.models or [],
            code_url=db_doc.code_url,
            open_access=db_doc.open_access,
            license=db_doc.license,
            provenance=prov,
            raw_source_metadata=db_doc.raw_source_metadata or {},
            retrieval_timestamp=db_doc.retrieval_timestamp,
        )

    # -------------------------------------------------------------------------
    # 2. Saved Papers
    # -------------------------------------------------------------------------
    async def save_paper(
        self,
        document_id: str,
        user_id: str = "default_user",
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> SavedPaperDB:
        stmt = select(SavedPaperDB).where(
            SavedPaperDB.document_id == document_id,
            SavedPaperDB.user_id == user_id,
        )
        res = await self.session.execute(stmt)
        existing = res.scalar_one_or_none()

        if existing:
            if notes is not None:
                existing.notes = notes
            if tags is not None:
                existing.tags = tags
            await self.session.commit()
            await self.session.refresh(existing)
            return existing

        new_entry = SavedPaperDB(
            document_id=document_id,
            user_id=user_id,
            notes=notes,
            tags=tags or [],
        )
        self.session.add(new_entry)
        await self.session.commit()
        await self.session.refresh(new_entry)
        return new_entry

    async def remove_saved_paper(self, document_id: str, user_id: str = "default_user") -> bool:
        stmt = delete(SavedPaperDB).where(
            SavedPaperDB.document_id == document_id,
            SavedPaperDB.user_id == user_id,
        )
        res = await self.session.execute(stmt)
        await self.session.commit()
        return res.rowcount > 0

    async def get_saved_papers(self, user_id: str = "default_user") -> List[SavedPaperDB]:
        stmt = select(SavedPaperDB).where(SavedPaperDB.user_id == user_id)
        res = await self.session.execute(stmt)
        return list(res.scalars().all())

    # -------------------------------------------------------------------------
    # 3. User Feedback
    # -------------------------------------------------------------------------
    async def save_feedback(self, item: FeedbackItem, user_id: str = "default_user") -> UserFeedbackDB:
        fb_db = UserFeedbackDB(
            id=str(uuid.uuid4()),
            document_id=item.document_id or "",
            feedback_type=item.feedback_type.value,
            target_document_id=item.preferred_doc_id or item.inferior_doc_id,
            rationale=item.notes,
            user_id=user_id,
        )
        self.session.add(fb_db)
        await self.session.commit()
        await self.session.refresh(fb_db)
        return fb_db

    async def get_feedback_items(self, user_id: str = "default_user") -> List[FeedbackItem]:
        stmt = select(UserFeedbackDB).where(UserFeedbackDB.user_id == user_id)
        res = await self.session.execute(stmt)
        rows = res.scalars().all()
        return [
            FeedbackItem(
                document_id=r.document_id,
                feedback_type=FeedbackType(r.feedback_type),
                preferred_doc_id=r.target_document_id,
                notes=r.rationale,
            )
            for r in rows
        ]

    # -------------------------------------------------------------------------
    # 4. User Preferences
    # -------------------------------------------------------------------------
    async def save_preferences(self, prefs: UserPreferences, user_id: str = "default_user") -> UserPreferencesDB:
        existing = await self.session.get(UserPreferencesDB, user_id)
        prefs_data = prefs.model_dump()
        if existing:
            existing.preferences = prefs_data
            pref_db = existing
        else:
            pref_db = UserPreferencesDB(user_id=user_id, preferences=prefs_data)
            self.session.add(pref_db)
        await self.session.commit()
        await self.session.refresh(pref_db)
        return pref_db

    async def get_preferences(self, user_id: str = "default_user") -> Optional[UserPreferences]:
        db_pref = await self.session.get(UserPreferencesDB, user_id)
        if not db_pref:
            return None
        return UserPreferences(**db_pref.preferences)

    # -------------------------------------------------------------------------
    # 5. Utility Runs
    # -------------------------------------------------------------------------
    async def save_utility_run(self, record: UtilityRunRecord) -> UtilityRunDB:
        rec_data = record.model_dump(mode="json")
        run_db = UtilityRunDB(
            run_id=record.run_id,
            weights=rec_data["weights"],
            preferences=rec_data["preferences"],
            feedback_items=rec_data["feedback_items"],
            constraints=rec_data["constraints"],
            total_candidates=record.total_candidates,
            admissible_count=record.admissible_count,
            evaluations=rec_data["evaluations"],
            created_at=record.created_at,
        )
        self.session.add(run_db)
        await self.session.commit()
        await self.session.refresh(run_db)
        return run_db

    async def get_utility_run(self, run_id: str) -> Optional[UtilityRunRecord]:
        run_db = await self.session.get(UtilityRunDB, run_id)
        if not run_db:
            return None
        return UtilityRunRecord(
            run_id=run_db.run_id,
            created_at=run_db.created_at,
            weights=UtilityWeights(**run_db.weights),
            preferences=UserPreferences(**run_db.preferences),
            feedback_items=[FeedbackItem(**fb) for fb in run_db.feedback_items],
            constraints=run_db.constraints,
            total_candidates=run_db.total_candidates,
            admissible_count=run_db.admissible_count,
            evaluations=[
                CandidateEvaluation(
                    document_id=e["document_id"],
                    title=e["title"],
                    scores=UtilityComponentScores(**e["scores"]),
                )
                for e in run_db.evaluations
            ],
        )

    # -------------------------------------------------------------------------
    # 6. Optimization Results
    # -------------------------------------------------------------------------
    async def save_optimization_result(
        self,
        result: OptimizationResult,
        run_id: Optional[str] = None,
    ) -> OptimizationResultDB:
        ranked_ids = [c.document.internal_id for c in result.ranked_candidates]
        opt_db = OptimizationResultDB(
            run_id=run_id,
            objective_value=result.objective_value,
            solver_used=result.solver_used.value,
            solver_status=result.solver_status.value,
            ranked_document_ids=ranked_ids,
            verification_report=result.verification.model_dump(mode="json"),
            guarantee_scope=result.guarantee_scope,
        )
        self.session.add(opt_db)
        await self.session.commit()
        await self.session.refresh(opt_db)
        return opt_db

    # -------------------------------------------------------------------------
    # 7. Research Roadmaps
    # -------------------------------------------------------------------------
    async def save_roadmap(self, roadmap: ResearchRoadmap) -> ResearchRoadmapDB:
        roadmap_data = roadmap.model_dump(mode="json")
        rm_db = ResearchRoadmapDB(
            roadmap_id=roadmap.roadmap_id,
            target_query=roadmap.target_query,
            title=roadmap.title,
            stages=roadmap_data["stages"],
            supporting_papers=roadmap.supporting_papers,
            total_duration=roadmap.total_estimated_duration,
        )
        self.session.add(rm_db)
        await self.session.commit()
        await self.session.refresh(rm_db)
        return rm_db

    async def get_roadmap(self, roadmap_id: str) -> Optional[ResearchRoadmap]:
        rm_db = await self.session.get(ResearchRoadmapDB, roadmap_id)
        if not rm_db:
            return None
        return ResearchRoadmap(
            roadmap_id=rm_db.roadmap_id,
            title=rm_db.title,
            target_query=rm_db.target_query,
            stages=rm_db.stages,
            supporting_papers=rm_db.supporting_papers or [],
            total_estimated_duration=rm_db.total_duration,
            created_at=rm_db.created_at.isoformat(),
        )
