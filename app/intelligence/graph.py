"""
Scholarly Knowledge Graph construction module for ResearchMind AI.

Extracts and links:
- Paper -> Dataset
- Paper -> Model
- Paper -> Method
- Paper -> Repository
- Paper -> Related Paper (Citations, References, Shared Concepts)

Preserves data provenance and origin traceability for all nodes and edges.
"""

import logging
import re
from typing import Dict, List, Set

from app.intelligence.models import (
    EdgeType,
    GraphEdge,
    GraphNode,
    GraphProvenance,
    KnowledgeGraph,
    NodeType,
)
from app.schemas.paper import ResearchDocument

logger = logging.getLogger("researchmind.intelligence.graph")


def _slugify(text: str) -> str:
    """Normalize text string into a clean identifier slug."""
    clean = re.sub(r"[^a-zA-Z0-9_\-]+", "_", text.strip().lower())
    return clean[:80] or "entity"


class KnowledgeGraphBuilder:
    """
    Constructs multi-entity scholarly knowledge graphs with relationship provenance.
    """

    def build_graph(self, documents: List[ResearchDocument]) -> KnowledgeGraph:
        """
        Build an interconnected knowledge graph from a collection of research documents.
        """
        nodes_dict: Dict[str, GraphNode] = {}
        edges_list: List[GraphEdge] = []
        seen_edges: Set[str] = set()

        doc_by_id: Dict[str, ResearchDocument] = {}
        doc_by_doi: Dict[str, str] = {}  # doi -> internal_id

        # Index papers
        for doc in documents:
            doc_by_id[doc.internal_id] = doc
            if doc.doi:
                doc_by_doi[doc.doi.lower().strip()] = doc.internal_id

        # -------------------------------------------------------------
        # 1. Create Paper Nodes and Linked Entities
        # -------------------------------------------------------------
        for doc in documents:
            paper_node_id = f"paper:{doc.internal_id}"
            if paper_node_id not in nodes_dict:
                paper_prov = GraphProvenance(
                    source_document_id=doc.internal_id,
                    source_provider=doc.source,
                    extracted_field="canonical_record",
                    confidence=1.0,
                )
                nodes_dict[paper_node_id] = GraphNode(
                    id=paper_node_id,
                    type=NodeType.PAPER,
                    label=doc.title,
                    properties={
                        "internal_id": doc.internal_id,
                        "year": doc.publication_year,
                        "citations": doc.citation_count,
                        "venue": doc.venue,
                        "doi": doc.doi,
                        "url": doc.url,
                    },
                    provenance=paper_prov,
                )

            # Paper -> Dataset
            for ds in (doc.datasets or []):
                ds_name = ds.strip()
                if not ds_name:
                    continue
                ds_id = f"dataset:{_slugify(ds_name)}"
                if ds_id not in nodes_dict:
                    nodes_dict[ds_id] = GraphNode(
                        id=ds_id,
                        type=NodeType.DATASET,
                        label=ds_name,
                        properties={"dataset_name": ds_name},
                        provenance=GraphProvenance(
                            source_document_id=doc.internal_id,
                            source_provider=doc.source,
                            extracted_field="datasets",
                            confidence=0.95,
                        ),
                    )
                edge_key = f"{paper_node_id}->{ds_id}:uses_dataset"
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges_list.append(
                        GraphEdge(
                            source=paper_node_id,
                            target=ds_id,
                            type=EdgeType.USES_DATASET,
                            label="uses dataset",
                            properties={"paper_title": doc.title, "dataset": ds_name},
                            provenance=GraphProvenance(
                                source_document_id=doc.internal_id,
                                source_provider=doc.source,
                                extracted_field="datasets",
                                confidence=0.95,
                            ),
                        )
                    )

            # Paper -> Model
            for m in (doc.models or []):
                m_name = m.strip()
                if not m_name:
                    continue
                m_id = f"model:{_slugify(m_name)}"
                if m_id not in nodes_dict:
                    nodes_dict[m_id] = GraphNode(
                        id=m_id,
                        type=NodeType.MODEL,
                        label=m_name,
                        properties={"model_name": m_name},
                        provenance=GraphProvenance(
                            source_document_id=doc.internal_id,
                            source_provider=doc.source,
                            extracted_field="models",
                            confidence=0.95,
                        ),
                    )
                edge_key = f"{paper_node_id}->{m_id}:uses_model"
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges_list.append(
                        GraphEdge(
                            source=paper_node_id,
                            target=m_id,
                            type=EdgeType.USES_MODEL,
                            label="uses model",
                            properties={"paper_title": doc.title, "model": m_name},
                            provenance=GraphProvenance(
                                source_document_id=doc.internal_id,
                                source_provider=doc.source,
                                extracted_field="models",
                                confidence=0.95,
                            ),
                        )
                    )

            # Paper -> Method
            for method in (doc.methods or []):
                meth_name = method.strip()
                if not meth_name:
                    continue
                meth_id = f"method:{_slugify(meth_name)}"
                if meth_id not in nodes_dict:
                    nodes_dict[meth_id] = GraphNode(
                        id=meth_id,
                        type=NodeType.METHOD,
                        label=meth_name,
                        properties={"method_name": meth_name},
                        provenance=GraphProvenance(
                            source_document_id=doc.internal_id,
                            source_provider=doc.source,
                            extracted_field="methods",
                            confidence=0.90,
                        ),
                    )
                edge_key = f"{paper_node_id}->{meth_id}:uses_method"
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges_list.append(
                        GraphEdge(
                            source=paper_node_id,
                            target=meth_id,
                            type=EdgeType.USES_METHOD,
                            label="employs method",
                            properties={"paper_title": doc.title, "method": meth_name},
                            provenance=GraphProvenance(
                                source_document_id=doc.internal_id,
                                source_provider=doc.source,
                                extracted_field="methods",
                                confidence=0.90,
                            ),
                        )
                    )

            # Paper -> Repository
            if doc.code_url and doc.code_url.strip():
                repo_url = doc.code_url.strip()
                repo_id = f"repo:{_slugify(repo_url)}"
                if repo_id not in nodes_dict:
                    nodes_dict[repo_id] = GraphNode(
                        id=repo_id,
                        type=NodeType.REPOSITORY,
                        label=repo_url,
                        properties={"url": repo_url},
                        provenance=GraphProvenance(
                            source_document_id=doc.internal_id,
                            source_provider=doc.source,
                            extracted_field="code_url",
                            confidence=1.0,
                        ),
                    )
                edge_key = f"{paper_node_id}->{repo_id}:has_repository"
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges_list.append(
                        GraphEdge(
                            source=paper_node_id,
                            target=repo_id,
                            type=EdgeType.HAS_REPOSITORY,
                            label="has repository",
                            properties={"url": repo_url},
                            provenance=GraphProvenance(
                                source_document_id=doc.internal_id,
                                source_provider=doc.source,
                                extracted_field="code_url",
                                confidence=1.0,
                            ),
                        )
                    )

        # -------------------------------------------------------------
        # 2. Paper -> Related Paper Relationships
        # -------------------------------------------------------------
        for doc in documents:
            source_paper_id = f"paper:{doc.internal_id}"

            # Direct citations / references within the document set
            for ref in (doc.references or []):
                ref_clean = ref.strip().lower()
                target_doc_id = None
                if ref in doc_by_id:
                    target_doc_id = ref
                elif ref_clean in doc_by_doi:
                    target_doc_id = doc_by_doi[ref_clean]

                if target_doc_id and target_doc_id != doc.internal_id:
                    target_paper_id = f"paper:{target_doc_id}"
                    edge_key = f"{source_paper_id}->{target_paper_id}:cites_paper"
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges_list.append(
                            GraphEdge(
                                source=source_paper_id,
                                target=target_paper_id,
                                type=EdgeType.CITES_PAPER,
                                label="cites",
                                properties={"reference_key": ref},
                                provenance=GraphProvenance(
                                    source_document_id=doc.internal_id,
                                    source_provider=doc.source,
                                    extracted_field="references",
                                    confidence=1.0,
                                ),
                            )
                        )

            # Conceptual similarity between pairs of papers
            for other_doc in documents:
                if other_doc.internal_id == doc.internal_id:
                    continue
                other_paper_id = f"paper:{other_doc.internal_id}"
                c1 = set(c.lower().strip() for c in (doc.concepts or []))
                c2 = set(c.lower().strip() for c in (other_doc.concepts or []))
                overlap = c1.intersection(c2)
                if len(overlap) >= 2:
                    edge_key_a = f"{source_paper_id}->{other_paper_id}:similar_to_paper"
                    edge_key_b = f"{other_paper_id}->{source_paper_id}:similar_to_paper"
                    if edge_key_a not in seen_edges and edge_key_b not in seen_edges:
                        seen_edges.add(edge_key_a)
                        edges_list.append(
                            GraphEdge(
                                source=source_paper_id,
                                target=other_paper_id,
                                type=EdgeType.SIMILAR_TO_PAPER,
                                label="shared concepts",
                                properties={"shared_concepts": sorted(list(overlap))},
                                provenance=GraphProvenance(
                                    source_document_id=doc.internal_id,
                                    source_provider="ontology",
                                    extracted_field="concepts",
                                    confidence=0.85,
                                ),
                            )
                        )

        # -------------------------------------------------------------
        # 3. Summary Statistics
        # -------------------------------------------------------------
        type_counts: Dict[str, int] = {}
        for node in nodes_dict.values():
            type_str = node.type.value
            type_counts[type_str] = type_counts.get(type_str, 0) + 1

        return KnowledgeGraph(
            nodes=list(nodes_dict.values()),
            edges=edges_list,
            total_nodes=len(nodes_dict),
            total_edges=len(edges_list),
            summary_by_type=type_counts,
        )
