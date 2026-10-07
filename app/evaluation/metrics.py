"""
Evaluation metrics module for ResearchMind AI (Phase 9).

Implements rigorous formal evaluation metrics:
- Precision@K
- Recall@K
- NDCG@K
- Search Success Rate
- User Satisfaction Score
- Optimization Verification Rate
- Buildability Evaluation against Ground Truth
"""

import math
from typing import Any, Dict, List, Optional, Set

from app.feedback.models import FeedbackItem, FeedbackType
from app.intelligence.models import BuildabilityReport
from app.optimization.models import VerificationReport


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Computes Precision@K:
    Precision@K = |Retrieved[:K] cap Relevant| / K
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return round(relevant_retrieved / float(k), 4)


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Computes Recall@K:
    Recall@K = |Retrieved[:K] cap Relevant| / |Relevant|
    """
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return round(relevant_retrieved / float(len(relevant_ids)), 4)


def dcg_at_k(retrieved_ids: List[str], relevance_map: Dict[str, float], k: int) -> float:
    """
    Discounted Cumulative Gain at rank K:
    DCG@K = sum_{j=1}^K (2^{rel_j} - 1) / log2(1 + j)
    """
    dcg = 0.0
    for j, doc_id in enumerate(retrieved_ids[:k]):
        rel = relevance_map.get(doc_id, 0.0)
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(2.0 + j)
        dcg += gain / discount
    return dcg


def ndcg_at_k(
    retrieved_ids: List[str],
    relevance_map: Any,
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain at rank K:
    NDCG@K = DCG@K / IDCG@K
    """
    if k <= 0 or not relevance_map:
        return 0.0

    if isinstance(relevance_map, (set, list, tuple)):
        rel_dict: Dict[str, float] = {doc_id: 1.0 for doc_id in relevance_map}
    else:
        rel_dict = dict(relevance_map)

    actual_dcg = dcg_at_k(retrieved_ids, rel_dict, k)

    # Compute Ideal DCG (IDCG) by sorting relevant documents descending by score
    sorted_ideal_ids = sorted(rel_dict.keys(), key=lambda x: rel_dict[x], reverse=True)
    ideal_dcg = dcg_at_k(sorted_ideal_ids, rel_dict, k)

    if ideal_dcg <= 0.0:
        return 0.0 if actual_dcg == 0.0 else 1.0

    return round(min(1.0, max(0.0, actual_dcg / ideal_dcg)), 4)


def search_success_rate(
    sessions_or_retrieved: Any,
    ground_truth_sets: Optional[List[Set[str]]] = None,
    k: int = 5,
) -> float:
    """
    Computes search success rate across query sessions.
    Supports either list of session outcome dicts or list of candidate ID rankings compared against ground truth.
    """
    if not sessions_or_retrieved:
        return 0.0

    if ground_truth_sets is not None:
        total = len(sessions_or_retrieved)
        successful = 0
        for ret, gt in zip(sessions_or_retrieved, ground_truth_sets):
            if any(doc in gt for doc in ret[:k]):
                successful += 1
        return round(successful / float(total), 4)

    successful = sum(
        1 for s in sessions_or_retrieved
        if isinstance(s, dict) and s.get("total_results", 0) > 0 and s.get("status", "ok") == "ok"
    )
    return round(successful / float(len(sessions_or_retrieved)), 4)


def user_satisfaction_score(feedback_items: List[FeedbackItem]) -> float:
    """
    Computes normalized user satisfaction score based on active user feedback signals.
    Positive feedback (RELEVANT, PREFER_SIMILAR) vs total feedback.
    """
    if not feedback_items:
        return 1.0  # Neutral default baseline

    positive = 0
    total = len(feedback_items)
    for fb in feedback_items:
        if fb.feedback_type in (FeedbackType.RELEVANT, FeedbackType.PREFER_SIMILAR):
            positive += 1
        elif fb.feedback_type == FeedbackType.PAIRWISE_PREFERENCE:
            positive += 1  # Informative preference count

    return round(positive / float(total), 4)


def optimization_verification_rate(verification_reports: List[VerificationReport]) -> float:
    """
    Computes the verification rate of global ranking optimizations.
    Proportion of runs where all hard constraints and objective recomputations verified.
    """
    if not verification_reports:
        return 1.0
    verified = sum(1 for r in verification_reports if r.is_verified and r.constraints_satisfied)
    return round(verified / float(len(verification_reports)), 4)


def evaluate_buildability_accuracy(
    predicted_reports: List[BuildabilityReport],
    ground_truth: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """
    Evaluate predicted buildability scores against ground truth verification data.
    
    Returns:
    - mean_absolute_error: Average discrepancy on overall score
    - tier_accuracy: Proportion of exact reproducibility tier matches
    """
    if not predicted_reports:
        return {"mean_absolute_error": 0.0, "tier_accuracy": 1.0, "evaluated_count": 0}

    total_error = 0.0
    tier_matches = 0
    count = 0

    for rep in predicted_reports:
        if rep.document_id in ground_truth:
            gt = ground_truth[rep.document_id]
            gt_score = gt.get("overall_score", rep.overall_score)
            gt_tier = gt.get("reproducibility_tier", rep.reproducibility_tier)

            total_error += abs(rep.overall_score - gt_score)
            if rep.reproducibility_tier.lower() == gt_tier.lower():
                tier_matches += 1
            count += 1

    if count == 0:
        return {"mean_absolute_error": 0.0, "tier_accuracy": 1.0, "evaluated_count": 0}

    return {
        "mean_absolute_error": round(total_error / float(count), 4),
        "tier_accuracy": round(tier_matches / float(count), 4),
        "evaluated_count": count,
    }
