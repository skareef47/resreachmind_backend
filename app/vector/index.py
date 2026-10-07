"""
Vector Index layer for ResearchMind AI (Phase 9).

Provides high-performance vector search, versioned metadata, and index rebuild capability.
Maintains exact cosine similarity over normalized dense embeddings with document ID mapping.
"""

from datetime import datetime, timezone
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from app.schemas.paper import ResearchDocument

logger = logging.getLogger("researchmind.vector.index")


class VectorIndex:
    """
    Persistent vector index managing document embeddings, model versions,
    index versions, and fast cosine similarity search.
    """

    def __init__(
        self,
        dimension: int = 384,
        embedding_model_version: str = "all-MiniLM-L6-v2-v1.0",
        index_version: int | str = 1,
        model_version: Optional[str] = None,
    ) -> None:
        self.dimension = dimension
        self.embedding_model_version = model_version or embedding_model_version
        self.index_version = str(index_version)
        self.doc_ids: List[str] = []
        self.embeddings: np.ndarray = np.zeros((0, dimension), dtype=np.float32)
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.updated_at: str = self.created_at

    @property
    def size(self) -> int:
        """Total vectors indexed."""
        return len(self.doc_ids)

    @property
    def model_version(self) -> str:
        """Alias for embedding_model_version."""
        return self.embedding_model_version

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2 normalize vector(s) for cosine similarity."""
        norm = np.linalg.norm(vec, axis=-1, keepdims=True)
        norm = np.where(norm == 0.0, 1.0, norm)
        return (vec / norm).astype(np.float32)

    def add(self, document_id: str, embedding: np.ndarray, model_version: Optional[str] = None) -> None:
        """Add a single document embedding."""
        if document_id in self.doc_ids:
            # Update existing
            idx = self.doc_ids.index(document_id)
            norm_vec = self._normalize(embedding.reshape(1, -1))
            self.embeddings[idx] = norm_vec[0]
        else:
            norm_vec = self._normalize(embedding.reshape(1, -1))
            self.doc_ids.append(document_id)
            if self.embeddings.shape[0] == 0:
                self.embeddings = norm_vec
            else:
                self.embeddings = np.vstack([self.embeddings, norm_vec])

        if model_version:
            self.embedding_model_version = model_version
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def add_batch(
        self,
        document_ids: List[str],
        embeddings: np.ndarray,
        model_version: Optional[str] = None,
    ) -> None:
        """Add or update multiple document embeddings."""
        if len(document_ids) == 0:
            return

        norm_vecs = self._normalize(embeddings)
        for doc_id, vec in zip(document_ids, norm_vecs):
            self.add(doc_id, vec, model_version)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search vector index by cosine similarity.
        Returns ordered list of (document_id, similarity_score).
        """
        if self.size == 0 or top_k <= 0:
            return []

        norm_q = self._normalize(query_embedding.reshape(1, -1))
        # Inner product of normalized vectors equals cosine similarity in [-1.0, 1.0]
        scores = np.dot(self.embeddings, norm_q.T).flatten()

        actual_k = min(top_k, len(self.doc_ids))
        top_indices = np.argsort(-scores)[:actual_k]

        results = [(self.doc_ids[idx], float(round(scores[idx], 4))) for idx in top_indices]
        return results

    def rebuild(
        self,
        documents: List[ResearchDocument],
        embedding_fn: Callable[[Any], np.ndarray],
        model_version: Optional[str] = None,
    ) -> None:
        """
        Rebuild the vector index from scratch using the provided documents and embedder.
        """
        target_model = model_version or self.embedding_model_version
        try:
            next_ver = str(int(self.index_version) + 1)
        except ValueError:
            next_ver = f"{self.index_version}.1"

        if not documents:
            self.doc_ids = []
            self.embeddings = np.zeros((0, self.dimension), dtype=np.float32)
            self.index_version = next_ver
            self.updated_at = datetime.now(timezone.utc).isoformat()
            return

        texts = [f"{d.title} {d.abstract or ''}".strip() for d in documents]
        doc_ids = [d.internal_id for d in documents]

        try:
            dense_matrix = embedding_fn(texts)
        except Exception:
            dense_matrix = np.array([embedding_fn(t) for t in texts])

        if dense_matrix.ndim == 1:
            dense_matrix = dense_matrix.reshape(1, -1)

        self.dimension = dense_matrix.shape[1]
        self.doc_ids = doc_ids
        self.embeddings = self._normalize(dense_matrix)
        self.embedding_model_version = target_model
        self.index_version = next_ver
        self.updated_at = datetime.now(timezone.utc).isoformat()
        logger.info(
            "Rebuilt vector index: %d vectors, dim=%d, model_version=%s, index_version=%s",
            len(self.doc_ids),
            self.dimension,
            self.embedding_model_version,
            self.index_version,
        )

    def save(self, directory_path: str) -> None:
        """Persist vector index and metadata to disk."""
        os.makedirs(directory_path, exist_ok=True)
        # 1. Embeddings array
        np.save(os.path.join(directory_path, "embeddings.npy"), self.embeddings)

        # 2. Metadata
        meta = {
            "dimension": self.dimension,
            "embedding_model_version": self.embedding_model_version,
            "index_version": self.index_version,
            "doc_ids": self.doc_ids,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "total_vectors": len(self.doc_ids),
        }
        with open(os.path.join(directory_path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory_path: str) -> "VectorIndex":
        """Load vector index and metadata from disk and return new VectorIndex instance."""
        meta_file = os.path.join(directory_path, "metadata.json")
        emb_file = os.path.join(directory_path, "embeddings.npy")
        if not os.path.exists(meta_file) or not os.path.exists(emb_file):
            raise FileNotFoundError(f"Vector index files not found in {directory_path}")

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        idx = cls(
            dimension=meta["dimension"],
            embedding_model_version=meta["embedding_model_version"],
            index_version=meta["index_version"],
        )
        idx.doc_ids = meta["doc_ids"]
        idx.created_at = meta["created_at"]
        idx.updated_at = meta["updated_at"]
        idx.embeddings = np.load(emb_file)
        return idx


# Singleton instance
_GLOBAL_VECTOR_INDEX = VectorIndex()


def get_vector_index() -> VectorIndex:
    """Dependency provider for global vector index."""
    return _GLOBAL_VECTOR_INDEX
