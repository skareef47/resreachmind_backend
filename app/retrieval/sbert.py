import hashlib
import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.schemas.paper import ResearchDocument


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two numeric vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm1 * norm2)))


class DeterministicSemanticEmbedding:
    """Deterministic dense embedding generator for semantic similarity testing and fallback."""

    def __init__(self, dimension: int = 128) -> None:
        self.dimension = dimension

    def encode_text(self, text: str) -> List[float]:
        """Project text into normalized dense embedding vector via n-gram hashing."""
        vector = [0.0] * self.dimension
        if not text:
            return vector

        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        if not tokens:
            return vector

        for token in tokens:
            # Word level feature
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if (h >> 7) & 1 else -1.0
            vector[idx] += sign * 1.5

            # Sub-word character trigrams for semantic/morphological sharing
            for i in range(len(token) - 2):
                ngram = token[i : i + 3]
                h_ng = int(hashlib.md5(ngram.encode("utf-8")).hexdigest(), 16)
                idx_ng = h_ng % self.dimension
                sign_ng = 1.0 if (h_ng >> 5) & 1 else -1.0
                vector[idx_ng] += sign_ng * 0.8

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0.0:
            return [x / norm for x in vector]
        return vector

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts into dense vector representations."""
        return [self.encode_text(t) for t in texts]


@dataclass
class SBERTRankedResult:
    """Individual ranked result from SBERT semantic scoring."""

    document: ResearchDocument
    score: float
    rank: int


class SBERTRetriever:
    """Configurable Sentence-BERT semantic embedding retriever."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dimension: int = 128,
        use_torch: bool = False,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.use_torch = use_torch
        self._hf_model = None

        if use_torch:
            try:
                from sentence_transformers import SentenceTransformer
                self._hf_model = SentenceTransformer(model_name)
            except ImportError:
                self._hf_model = None

        self._fallback = DeterministicSemanticEmbedding(dimension=self.dimension)

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embeddings for a list of texts."""
        if self._hf_model is not None:
            embeddings = self._hf_model.encode(texts, convert_to_numpy=True)
            return [emb.tolist() for emb in embeddings]
        return self._fallback.encode(texts)

    def _build_doc_text(self, doc: ResearchDocument) -> str:
        """Create rich semantic passage from title, abstract, and concepts."""
        parts: List[str] = []
        if doc.title:
            parts.append(doc.title)
        if doc.abstract:
            parts.append(doc.abstract)
        if doc.concepts:
            parts.append(" ".join(doc.concepts))
        if doc.fields_of_study:
            parts.append(" ".join(doc.fields_of_study))
        return " ".join(parts)

    def rank(
        self,
        query: str,
        documents: List[ResearchDocument],
        top_k: Optional[int] = None,
    ) -> List[SBERTRankedResult]:
        """Rank corpus of ResearchDocuments against query using semantic cosine similarity."""
        if not documents:
            return []

        doc_texts = [self._build_doc_text(d) for d in documents]
        query_vec = self.encode([query])[0]
        doc_vecs = self.encode(doc_texts)

        scored: List[Tuple[float, int, ResearchDocument]] = []
        for idx, (doc, doc_vec) in enumerate(zip(documents, doc_vecs)):
            sim = cosine_similarity(query_vec, doc_vec)
            scored.append((sim, idx, doc))

        # Sort descending by cosine similarity, preserve document order on ties
        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

        limit = top_k if top_k is not None else len(documents)
        results: List[SBERTRankedResult] = []
        for rank_idx, (score, _, doc) in enumerate(scored[:limit], start=1):
            results.append(
                SBERTRankedResult(
                    document=doc,
                    score=round(score, 4),
                    rank=rank_idx,
                )
            )

        return results
