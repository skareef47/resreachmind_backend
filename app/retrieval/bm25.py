import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from app.schemas.paper import ResearchDocument

# Tokenizer pattern preserving DOIs, hyphenated technical terms, acronyms, and identifiers
TOKEN_PATTERN = re.compile(
    r"(?:"
    r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+|"       # DOI / identifier
    r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+|"     # Compound technical term (e.g. text-to-image, cross-encoder)
    r"[A-Z]{2,}(?![a-z])|"                    # Acronyms (e.g. BERT, RAG, LLM, CNN)
    r"[A-Za-z0-9]+|"                          # Standard words & numbers
    r")",
)


def tokenize_text(text: Optional[str]) -> List[str]:
    """Tokenize text preserving technical terms, rare tokens, acronyms, and DOIs."""
    if not text or not isinstance(text, str):
        return []

    raw_matches = TOKEN_PATTERN.findall(text)
    tokens: List[str] = []
    for match in raw_matches:
        lowered = match.lower()
        tokens.append(lowered)

        # For hyphenated terms, also index sub-words to support partial & exact match
        if "-" in match or "_" in match:
            sub_parts = re.split(r"[-_]", match)
            for sub in sub_parts:
                if len(sub) > 1:
                    tokens.append(sub.lower())

    return tokens


@dataclass
class BM25RankedResult:
    """Individual ranked result from BM25 scoring."""

    document: ResearchDocument
    score: float
    rank: int


class BM25Retriever:
    """Okapi BM25 ranking model for lexical paper retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def _build_doc_corpus(self, doc: ResearchDocument) -> str:
        """Combine document text fields with title weighting for lexical index."""
        parts: List[str] = []
        # Title repeated for term frequency boost
        if doc.title:
            parts.extend([doc.title, doc.title])
        if doc.abstract:
            parts.append(doc.abstract)
        if doc.doi:
            parts.append(doc.doi)
        if doc.venue:
            parts.append(doc.venue)
        if doc.fields_of_study:
            parts.extend(doc.fields_of_study)
        if doc.concepts:
            parts.extend(doc.concepts)
        return " ".join(parts)

    def rank(
        self,
        query: str,
        documents: List[ResearchDocument],
        top_k: Optional[int] = None,
    ) -> List[BM25RankedResult]:
        """Rank corpus of ResearchDocuments against query using Okapi BM25."""
        if not documents:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return [
                BM25RankedResult(document=doc, score=0.0, rank=idx + 1)
                for idx, doc in enumerate(documents)
            ]

        n_docs = len(documents)
        tokenized_docs: List[List[str]] = [
            tokenize_text(self._build_doc_corpus(doc)) for doc in documents
        ]
        doc_lengths: List[int] = [len(tokens) for tokens in tokenized_docs]
        avgdl = sum(doc_lengths) / max(n_docs, 1)

        # Compute document frequency for each token
        df: Dict[str, int] = Counter()
        doc_counters: List[Counter[str]] = []
        for tokens in tokenized_docs:
            c = Counter(tokens)
            doc_counters.append(c)
            for term in c.keys():
                df[term] += 1

        # Compute Robertson-Spärck Jones IDF with smoothing
        idf_cache: Dict[str, float] = {}
        for token in set(query_tokens):
            n_t = df.get(token, 0)
            # Okapi BM25 smoothed IDF formula
            idf = math.log((n_docs - n_t + 0.5) / (n_t + 0.5) + 1.0)
            idf_cache[token] = max(idf, 0.0)

        # Score documents
        scored: List[Tuple[float, int, ResearchDocument]] = []
        for idx, doc in enumerate(documents):
            score = 0.0
            doc_len = doc_lengths[idx]
            counter = doc_counters[idx]
            len_norm = 1.0 - self.b + self.b * (doc_len / (avgdl or 1.0))

            for q_term in query_tokens:
                if q_term not in counter:
                    continue
                tf = counter[q_term]
                idf = idf_cache.get(q_term, 0.0)
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * len_norm
                score += idf * (numerator / denominator)

            scored.append((score, idx, doc))

        # Sort descending by BM25 score, preserve document index on ties
        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

        limit = top_k if top_k is not None else n_docs
        results: List[BM25RankedResult] = []
        for rank_idx, (score, _, doc) in enumerate(scored[:limit], start=1):
            results.append(
                BM25RankedResult(
                    document=doc,
                    score=round(score, 4),
                    rank=rank_idx,
                )
            )

        return results
