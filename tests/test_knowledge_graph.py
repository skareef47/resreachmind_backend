"""
Unit and Integration Tests for ResearchMind AI Scholarly Knowledge Graph.

Covers Requirements:
- Graph generated from real candidate documents
- Nodes are strictly unique
- Edges reference only valid existing nodes
- No fabricated relationships (all edges carry factual evidence & confidence)
- Graph changes when query/result set changes
- API schema works (GET /api/v1/knowledge-graph and POST /api/v1/knowledge-graph)
- Search API returns knowledge_graph in payload
- Frontend consumes knowledge graph response structure
"""

import pytest
from fastapi.testclient import TestClient

from app.intelligence.graph import KnowledgeGraphBuilder
from app.intelligence.models import EdgeType, KnowledgeGraph, NodeType
from app.main import app
from app.schemas.paper import ResearchDocument

client = TestClient(app)


def make_test_doc(
    doc_id: str,
    title: str,
    abstract: str = "",
    authors: list = None,
    year: int = 2024,
    venue: str = "NeurIPS",
    concepts: list = None,
    datasets: list = None,
    models: list = None,
    methods: list = None,
    code_url: str = None,
    doi: str = None,
    references: list = None,
) -> ResearchDocument:
    return ResearchDocument(
        internal_id=f"doc_{doc_id}",
        source="openalex",
        source_id=doc_id,
        title=title,
        abstract=abstract,
        authors=authors or ["Alice Smith", "Bob Jones"],
        publication_year=year,
        venue=venue,
        concepts=concepts or ["Machine Learning", "Attention Mechanism"],
        datasets=datasets or [],
        models=models or [],
        methods=methods or [],
        code_url=code_url,
        doi=doi or f"10.1000/{doc_id}",
        references=references or [],
    )


def test_graph_generated_from_real_candidates():
    """Graph is properly built from candidate documents with multi-entity types."""
    doc1 = make_test_doc(
        "p1",
        "Attention and Memory in Transformer Architectures",
        abstract="We evaluate Transformer models on GLUE benchmark using Multi-Head Attention for Information Retrieval.",
        authors=["Alice Smith", "Charlie Brown"],
        venue="NeurIPS",
        concepts=["Deep Learning", "Information Retrieval"],
        datasets=["GLUE"],
        models=["Transformer"],
        methods=["Multi-Head Attention"],
        code_url="https://github.com/alice/transformer-attention",
    )
    doc2 = make_test_doc(
        "p2",
        "Graph Neural Networks for Entity Typing",
        abstract="Employs Graph Neural Network architecture evaluated on ImageNet for Entity Typing.",
        authors=["David Lee", "Alice Smith"],  # Alice Smith shared author!
        venue="ICML",
        concepts=["Graph Neural Network", "Entity Typing"],
        datasets=["ImageNet"],
        models=["GNN"],
        methods=["Backpropagation"],
    )

    builder = KnowledgeGraphBuilder()
    graph = builder.build_graph([doc1, doc2])

    assert isinstance(graph, KnowledgeGraph)
    assert graph.total_nodes > 0
    assert graph.total_edges > 0
    assert len(graph.nodes) == graph.total_nodes
    assert len(graph.edges) == graph.total_edges

    # Check that expected node types are present
    node_types = set(n.type.value for n in graph.nodes)
    assert "paper" in node_types
    assert "author" in node_types
    assert "venue" in node_types
    assert "concept" in node_types


def test_nodes_are_unique():
    """All node IDs in the synthesized knowledge graph must be unique."""
    doc1 = make_test_doc("u1", "Paper One", authors=["Alice Smith", "Bob Jones"], venue="NeurIPS")
    doc2 = make_test_doc("u2", "Paper Two", authors=["Alice Smith", "Charlie Lee"], venue="NeurIPS")

    builder = KnowledgeGraphBuilder()
    graph = builder.build_graph([doc1, doc2])

    node_ids = [n.id for n in graph.nodes]
    assert len(node_ids) == len(set(node_ids)), "Duplicate node IDs detected in knowledge graph!"


def test_edges_reference_valid_nodes():
    """Every edge's source and target must exist in the node set."""
    doc1 = make_test_doc(
        "v1",
        "Linear Transformer Framework",
        abstract="Evaluated on GLUE benchmark with ResNet backbone for Sentiment Analysis.",
        authors=["Bob Jones"],
        venue="ACL",
        datasets=["GLUE"],
        models=["ResNet"],
    )
    doc2 = make_test_doc("v2", "Convolutional Baselines", authors=["Carol White"], venue="CVPR")

    builder = KnowledgeGraphBuilder()
    graph = builder.build_graph([doc1, doc2])

    node_ids = set(n.id for n in graph.nodes)
    for edge in graph.edges:
        assert edge.source in node_ids, f"Edge source '{edge.source}' not found in graph nodes!"
        assert edge.target in node_ids, f"Edge target '{edge.target}' not found in graph nodes!"


def test_no_fabricated_relationships():
    """Every relationship must contain non-empty evidence, provenance, and positive confidence."""
    doc = make_test_doc(
        "e1",
        "Verified Attention Mechanisms",
        abstract="We use PyTorch to train BERT on SQuAD for Question Answering.",
        authors=["Elena Rostova"],
        venue="EMNLP",
        datasets=["SQuAD"],
        models=["BERT"],
        code_url="https://github.com/elena/verified-attention",
    )

    builder = KnowledgeGraphBuilder()
    graph = builder.build_graph([doc])

    assert len(graph.edges) > 0
    for edge in graph.edges:
        assert edge.evidence and len(edge.evidence.strip()) > 0, f"Edge {edge} has missing evidence!"
        assert 0.0 <= edge.confidence <= 1.0, f"Edge confidence {edge.confidence} out of range [0, 1]!"
        assert edge.provenance is not None, f"Edge {edge} lacks provenance!"


def test_graph_changes_when_result_set_changes():
    """Graph structure changes deterministically when candidate documents change."""
    doc_nlp = make_test_doc(
        "nlp1",
        "NLP Advances with Transformers",
        abstract="We study RoBERTa on SuperGLUE for Text Summarization.",
        authors=["Noam Shazeer"],
        venue="ACL",
        models=["RoBERTa"],
        datasets=["SuperGLUE"],
    )
    doc_vision = make_test_doc(
        "vis1",
        "Vision Transformers in Radiology",
        abstract="We study ViT on ImageNet for Object Detection.",
        authors=["Alexey Dosovitskiy"],
        venue="CVPR",
        models=["ViT"],
        datasets=["ImageNet"],
    )

    builder = KnowledgeGraphBuilder()
    graph_nlp = builder.build_graph([doc_nlp])
    graph_vision = builder.build_graph([doc_vision])

    nlp_labels = set(n.label for n in graph_nlp.nodes)
    vision_labels = set(n.label for n in graph_vision.nodes)

    assert "Noam Shazeer" in nlp_labels
    assert "Noam Shazeer" not in vision_labels
    assert "Alexey Dosovitskiy" in vision_labels
    assert "Alexey Dosovitskiy" not in nlp_labels


def test_api_knowledge_graph_post_with_documents():
    """POST /api/v1/knowledge-graph constructs graph from explicit document payload."""
    doc = make_test_doc("api1", "API Test Document", authors=["Grace Hopper"], venue="IEEE")
    payload = {
        "documents": [doc.model_dump()],
    }
    response = client.post("/api/v1/knowledge-graph", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "nodes" in data
    assert "edges" in data
    assert "total_nodes" in data
    assert "total_edges" in data
    assert any(n["label"] == "Grace Hopper" for n in data["nodes"])


def test_api_knowledge_graph_get_endpoint():
    """GET /api/v1/knowledge-graph?query=... retrieves live papers and returns graph."""
    response = client.get("/api/v1/knowledge-graph?query=transformer+attention&limit=3")
    assert response.status_code == 200
    data = response.json()

    assert "nodes" in data
    assert "edges" in data
    assert isinstance(data["nodes"], list)
    assert isinstance(data["edges"], list)
    if len(data["nodes"]) > 0:
        node_ids = set(n["id"] for n in data["nodes"])
        for edge in data["edges"]:
            assert edge["source"] in node_ids
            assert edge["target"] in node_ids


def test_search_api_returns_knowledge_graph():
    """POST /api/v1/search includes knowledge_graph in response payload."""
    payload = {
        "query": "transformer attention mechanism",
        "limit": 3,
        "sources": ["openalex"],
        "optimize": True,
    }
    response = client.post("/api/v1/search", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "knowledge_graph" in data
    assert data["knowledge_graph"] is not None
    kg = data["knowledge_graph"]
    assert "nodes" in kg
    assert "edges" in kg
    assert "total_nodes" in kg
    assert "total_edges" in kg


def test_frontend_consumes_graph_response():
    """Frontend can parse all nodes, edges, filter by type, and resolve connections."""
    doc = make_test_doc(
        "front1",
        "Frontend Compatibility Paper",
        abstract="Uses BERT on SQuAD for Information Retrieval.",
        authors=["Ada Lovelace"],
        venue="Nature",
        datasets=["SQuAD"],
        models=["BERT"],
    )
    builder = KnowledgeGraphBuilder()
    graph = builder.build_graph([doc])
    graph_dict = graph.model_dump()

    # Simulate frontend consumption
    nodes = graph_dict["nodes"]
    edges = graph_dict["edges"]
    assert len(nodes) > 0
    assert len(edges) > 0

    # Ensure every node has id, type, label, and metadata
    for n in nodes:
        assert "id" in n
        assert "type" in n
        assert "label" in n
        assert "metadata" in n

    # Simulate selecting a paper node and finding its connected relationships
    paper_node = next(n for n in nodes if n["type"] == "paper")
    connected = [e for e in edges if e["source"] == paper_node["id"] or e["target"] == paper_node["id"]]
    assert len(connected) > 0

    for e in connected:
        assert "evidence" in e
        assert "confidence" in e
        assert "type" in e
        assert len(e["evidence"]) > 0
