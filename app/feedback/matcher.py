import re
from typing import Dict, List, Optional, Set
from app.feedback.models import FeedbackItem, FeedbackType
from app.schemas.paper import ResearchDocument
from app.schemas.retrieval import RetrievedCandidate


def _tokenize(text: str) -> List[str]:
    """Extract lowercased alphanumeric tokens from text."""
    return [t.lower() for t in re.findall(r"\b[A-Za-z0-9\-]+\b", text)]



def _compute_token_jaccard(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute token Jaccard similarity between two token lists."""
    if not tokens1 or not tokens2:
        return 0.0
    s1, s2 = set(tokens1), set(tokens2)
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return float(intersection) / float(union) if union > 0 else 0.0


class FeedbackMatcher:
    """
    Evaluates how closely a candidate research paper aligns with active user feedback.
    
    Signals:
    - Direct pointwise endorsement (+1.0) or rejection (-1.0)
    - Semantic / lexical similarity to preferred documents (+sim)
    - Semantic / lexical similarity to excluded documents (-sim)
    - Pairwise preference satisfaction (+0.5 for preferred, -0.5 for inferior)
    """

    def __init__(
        self,
        reference_corpus: Optional[Dict[str, ResearchDocument]] = None,
    ) -> None:
        self.reference_corpus = reference_corpus or {}

    def compute_feedback_match(
        self,
        candidate: RetrievedCandidate,
        feedback_items: List[FeedbackItem],
    ) -> float:
        """
        Compute normalized feedback match score in [0.0, 1.0].
        
        Neutral baseline is 0.5 when no feedback impacts the candidate.
        Direct relevant marks push the score toward 1.0, while not-relevant
        marks push the score toward 0.0.
        """
        if not feedback_items:
            return 0.5  # Neutral default when no feedback exists

        doc = candidate.document
        doc_ids = {doc.internal_id, doc.source_id}
        if doc.doi:
            doc_ids.add(doc.doi)

        net_score = 0.0
        applied_weights = 0.0

        cand_text = f"{doc.title} {doc.abstract or ''}"
        cand_tokens = _tokenize(cand_text)

        for item in feedback_items:
            w = item.weight

            # 1. Direct Pointwise Feedback
            if item.feedback_type == FeedbackType.RELEVANT:
                if item.document_id and item.document_id.strip() in doc_ids:
                    net_score += 1.0 * w
                    applied_weights += w

            elif item.feedback_type == FeedbackType.NOT_RELEVANT:
                if item.document_id and item.document_id.strip() in doc_ids:
                    net_score -= 1.0 * w
                    applied_weights += w

            # 2. Similarity Feedback
            elif item.feedback_type == FeedbackType.PREFER_SIMILAR:
                target_id = item.document_id.strip() if item.document_id else ""
                ref_doc = self.reference_corpus.get(target_id)
                if ref_doc:
                    ref_tokens = _tokenize(f"{ref_doc.title} {ref_doc.abstract or ''}")
                    sim = _compute_token_jaccard(cand_tokens, ref_tokens)
                    net_score += sim * w
                    applied_weights += w * 0.5

            elif item.feedback_type == FeedbackType.EXCLUDE_SIMILAR:
                target_id = item.document_id.strip() if item.document_id else ""
                ref_doc = self.reference_corpus.get(target_id)
                if ref_doc:
                    ref_tokens = _tokenize(f"{ref_doc.title} {ref_doc.abstract or ''}")
                    sim = _compute_token_jaccard(cand_tokens, ref_tokens)
                    net_score -= sim * w
                    applied_weights += w * 0.5

            # 3. Pairwise Preferences
            elif item.feedback_type == FeedbackType.PAIRWISE_PREFERENCE:
                pref = item.preferred_doc_id.strip() if item.preferred_doc_id else ""
                inf = item.inferior_doc_id.strip() if item.inferior_doc_id else ""
                if pref in doc_ids:
                    net_score += 0.5 * w
                    applied_weights += w * 0.5
                elif inf in doc_ids:
                    net_score -= 0.5 * w
                    applied_weights += w * 0.5

        if applied_weights == 0.0:
            return 0.5

        # Bound and shift net_score to [0.0, 1.0]
        normalized = 0.5 + (net_score / (2.0 * max(1.0, applied_weights)))
        return round(max(0.0, min(1.0, normalized)), 4)
