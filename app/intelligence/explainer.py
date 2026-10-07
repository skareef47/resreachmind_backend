"""
Explainability engine for ResearchMind AI.

Generates transparent ranking explanations ONLY from actual stored ranking inputs
and run telemetry (UtilityRunRecord).
"""

import logging
from typing import Any, Dict, List, Optional

from app.intelligence.models import ComponentContribution, RankingExplanation
from app.optimization.utility import (
    CandidateEvaluation,
    UtilityComponentScores,
    UtilityRunRecord,
    UtilityRunStore,
    UtilityWeights,
    get_utility_run_store,
)

logger = logging.getLogger("researchmind.intelligence.explainer")


class RankingExplainer:
    """
    Transparent ranking explainer.
    
    Strictly derives explanations from the stored run audit records,
    guaranteeing that explanations reflect true algorithmic decisions
    rather than post-hoc hallucinated narratives.
    """

    def __init__(self, run_store: Optional[UtilityRunStore] = None) -> None:
        self.run_store = run_store or get_utility_run_store()

    def explain_candidate(
        self,
        run_record: UtilityRunRecord,
        document_id: str,
        rank: Optional[int] = None,
    ) -> RankingExplanation:
        """
        Generate a verifiable explanation for a specific document from a run record.
        """
        # Find candidate evaluation in run record
        eval_item: Optional[CandidateEvaluation] = None
        for item in run_record.evaluations:
            if item.document_id == document_id:
                eval_item = item
                break

        if eval_item is None:
            raise ValueError(
                f"Document '{document_id}' not found in ranking run '{run_record.run_id}' evaluations"
            )

        scores: UtilityComponentScores = eval_item.scores
        w: UtilityWeights = run_record.weights

        # 1. Component decompositions
        raw_components = [
            ("relevance", w.w1_relevance, scores.relevance, "Relevance score derived from hybrid retrieval and Cross-Encoder"),
            ("preference_match", w.w2_preference_match, scores.preference_match, "Alignment with active multi-dimensional user preferences"),
            ("feedback_match", w.w3_feedback_match, scores.feedback_match, "Alignment with active pointwise and similarity feedback signals"),
            ("buildability", w.w4_buildability, scores.buildability, "Engineering buildability proxy based on code, datasets, and models"),
            ("source_quality", w.w5_source_quality, scores.source_quality, "Scholarly prestige based on log-citations and venue prominence"),
            ("recency", w.w6_recency, scores.recency, "Publication recency based on exponential time decay"),
        ]

        # Calculate positive sum for percentage attribution
        positive_sum = sum(weight * score for _, weight, score, _ in raw_components)
        if positive_sum <= 0.0:
            positive_sum = 1.0

        components_dict: Dict[str, ComponentContribution] = {}
        for name, weight, score, desc in raw_components:
            weighted = round(weight * score, 6)
            pct = round((weighted / positive_sum) * 100.0, 2)
            components_dict[name] = ComponentContribution(
                name=name,
                weight=round(weight, 4),
                raw_score=round(score, 4),
                weighted_score=weighted,
                percentage_contribution=pct,
                description=desc,
            )

        # Penalties component
        penalties_val = round(scores.penalties, 6)
        components_dict["penalties"] = ComponentContribution(
            name="penalties",
            weight=1.0,
            raw_score=penalties_val,
            weighted_score=-penalties_val,
            percentage_contribution=0.0,
            description="Soft constraint deductions applied to candidate utility",
        )

        # 2. Summarize active preferences
        pref_dict: Dict[str, Any] = {}
        prefs = run_record.preferences
        for field in [
            "importance_relevance", "importance_implementation", "importance_recency",
            "compute_preference", "dataset_importance", "code_importance",
            "pretrained_model_importance", "source_quality_importance"
        ]:
            val = getattr(prefs, field, None)
            if val is not None:
                pref_dict[field] = getattr(val, "value", val)

        # 3. Summarize active feedback relevant to this document
        fb_summary: List[str] = []
        for fb in run_record.feedback_items:
            target_ids = {fb.document_id, getattr(fb, "preferred_doc_id", None), getattr(fb, "inferior_doc_id", None)}
            if document_id in target_ids:
                fb_summary.append(f"Direct feedback: {fb.feedback_type.value}")
            elif fb.feedback_type.value in ("prefer_similar", "exclude_similar"):
                fb_summary.append(f"Global similarity feedback active for: {fb.document_id}")

        # 4. Summarize applied penalties
        penalties_summary: Dict[str, float] = {}
        if penalties_val > 0.0:
            penalties_summary["total_penalties"] = penalties_val

        # 5. Grounded Narrative Explanation
        narrative_parts = [
            f"Paper '{eval_item.title}' achieved an overall utility score of {scores.utility_score:.4f}"
            + (f" (Rank {rank})." if rank is not None else ".")
        ]

        # Top driving components
        sorted_pos = sorted(
            [c for c in components_dict.values() if c.name != "penalties"],
            key=lambda x: x.weighted_score,
            reverse=True,
        )
        top_driver = sorted_pos[0]
        narrative_parts.append(
            f"The primary driver is {top_driver.name.replace('_', ' ')} contributing {top_driver.percentage_contribution:.1f}% "
            f"(raw score: {top_driver.raw_score:.3f}, weighted: {top_driver.weighted_score:.4f})."
        )

        # Buildability and Code status
        if scores.buildability >= 0.5:
            narrative_parts.append(
                f"Engineering buildability provided strong support (+{components_dict['buildability'].percentage_contribution:.1f}%), "
                f"reflecting available code, model checkpoints, or datasets."
            )
        else:
            narrative_parts.append(
                f"Buildability provided limited contribution (+{components_dict['buildability'].percentage_contribution:.1f}%), "
                f"indicating missing open-source code repositories or pretrained assets."
            )

        # Feedback impact
        if fb_summary:
            narrative_parts.append("User feedback actively shaped the score: " + "; ".join(fb_summary) + ".")

        # Penalties impact
        if penalties_val > 0.0:
            narrative_parts.append(
                f"A deduction of -{penalties_val:.4f} utility was applied due to soft constraint penalties."
            )

        narrative = " ".join(narrative_parts)

        return RankingExplanation(
            document_id=document_id,
            title=eval_item.title,
            run_id=run_record.run_id,
            rank=rank,
            final_utility_score=scores.utility_score,
            components=components_dict,
            narrative_explanation=narrative,
            active_preferences_summary=pref_dict,
            active_feedback_summary=fb_summary,
            applied_penalties_summary=penalties_summary,
            provenance_run_timestamp=run_record.created_at.isoformat(),
        )

    def explain_run(self, run_id: str) -> List[RankingExplanation]:
        """
        Generate explanations for all candidates in a stored ranking run.
        """
        record = self.run_store.get_run(run_id)
        if record is None:
            raise KeyError(f"Ranking run '{run_id}' not found in audit store")

        explanations: List[RankingExplanation] = []
        for rank_idx, eval_item in enumerate(record.evaluations):
            exp = self.explain_candidate(
                run_record=record,
                document_id=eval_item.document_id,
                rank=rank_idx + 1 if eval_item.scores.is_admissible else None,
            )
            explanations.append(exp)

        return explanations
