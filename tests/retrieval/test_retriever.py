from datetime import datetime, timezone

import pytest

from graphtool.chunking.types import Chunk
from graphtool.graph.types import Edge, GraphMetadata, KnowledgeGraph, Node
from graphtool.retrieval import SqliteChunkEmbeddingStore
from graphtool.retrieval.retriever import (
    prepare_chunk_retriever,
    retrieve_context,
)
from graphtool.storage import open_database


class FakeEmbeddingClient:
    embedding_model = "fake-embedding-model"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[str] = []
        self.batch_calls: list[list[str]] = []

    def embed_texts(self, texts) -> list[list[float]]:
        batch = list(texts)
        self.calls.extend(batch)
        self.batch_calls.append(batch)
        return [self._vector_for(text) for text in batch]

    def _vector_for(self, text: str) -> list[float]:
        for marker, vector in self.vectors.items():
            if marker in text:
                return vector
        return [0.0, 1.0]


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            id="doc-chunk-0000",
            source="doc.md",
            index=0,
            text="# Python\nPython is a programming language.",
            heading_path=["Python"],
        ),
        Chunk(
            id="doc-chunk-0001",
            source="doc.md",
            index=1,
            text="## Pydantic\nPydantic is a validation library built for Python.",
            heading_path=["Python", "Pydantic"],
        ),
        Chunk(
            id="doc-chunk-0002",
            source="doc.md",
            index=2,
            text="## FastAPI\nFastAPI uses Pydantic for request validation.",
            heading_path=["Python", "FastAPI"],
        ),
        Chunk(
            id="doc-chunk-0003",
            source="doc.md",
            index=3,
            text="## Django\nDjango includes an ORM.",
            heading_path=["Python", "Django"],
        ),
    ]


def _graph() -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes=[
            Node(
                id="python",
                label="Python",
                type="Language",
                chunk_ids=["doc-chunk-0000", "doc-chunk-0001"],
            ),
            Node(
                id="pydantic",
                label="Pydantic",
                type="Library",
                suggested_type="schema validator",
                aliases=["data parser"],
                properties={"purpose": "data validation"},
                chunk_ids=["doc-chunk-0001", "missing-chunk"],
            ),
            Node(
                id="fastapi",
                label="FastAPI",
                type="Framework",
                chunk_ids=["doc-chunk-0002"],
            ),
            Node(
                id="django",
                label="Django",
                type="Framework",
                chunk_ids=["doc-chunk-0003"],
            ),
        ],
        edges=[
            Edge(
                id="edge-0001",
                source="pydantic",
                target="python",
                label="built_for",
                properties={"compatibility": "Python 3.13"},
                chunk_ids=["doc-chunk-0001"],
            ),
            Edge(
                id="edge-0002",
                source="fastapi",
                target="pydantic",
                label="uses",
                chunk_ids=["doc-chunk-0002"],
            ),
            Edge(
                id="edge-0003",
                source="django",
                target="python",
                label="includes",
                chunk_ids=["doc-chunk-0003"],
            ),
        ],
        metadata=GraphMetadata(
            source="doc.md",
            content_hash="hash",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    )


@pytest.mark.parametrize(
    "query",
    [
        "data parser",
        "schema validator",
        "purpose data validation",
    ],
)
def test_retrieve_context_bm25_searches_enriched_node_fields(query):
    chunks = [
        Chunk(
            id="target",
            source="doc.md",
            index=0,
            text="Ordinary content.",
        ),
        Chunk(
            id="other",
            source="doc.md",
            index=1,
            text="Unrelated billing content.",
        ),
    ]
    graph = KnowledgeGraph(
        nodes=[
            Node(
                id="entity",
                label="Pydantic",
                type="Library",
                suggested_type="schema validator",
                aliases=["data parser"],
                properties={"purpose": "data validation"},
                chunk_ids=["target"],
            )
        ],
        edges=[],
    )

    result = retrieve_context(query, graph, chunks, top_chunks=1)

    assert [hit.chunk.id for hit in result.chunks] == ["target"]
    assert [node.id for node in result.chunks[0].linked_nodes] == ["entity"]


def test_retrieve_context_prioritizes_entity_label_over_heading():
    chunks = [
        Chunk(
            id="entity-label",
            source="doc.md",
            index=0,
            text="General product overview.",
            heading_path=["Products"],
        ),
        Chunk(
            id="heading-only",
            source="doc.md",
            index=1,
            text="General configuration overview.",
            heading_path=["Claude Code"],
        ),
    ]
    graph = KnowledgeGraph(
        nodes=[
            Node(
                id="claude-code",
                label="Claude Code",
                type="Product",
                chunk_ids=["entity-label"],
            )
        ],
        edges=[],
    )

    result = retrieve_context("Claude Code", graph, chunks)

    assert [hit.chunk.id for hit in result.chunks] == [
        "entity-label",
        "heading-only",
    ]
    assert result.chunks[0].score > result.chunks[1].score


def test_retrieve_context_prioritizes_alias_over_content():
    chunks = [
        Chunk(
            id="alias",
            source="doc.md",
            index=0,
            text="General command overview.",
        ),
        Chunk(
            id="content-only",
            source="doc.md",
            index=1,
            text="Conversation reset instructions.",
        ),
    ]
    graph = KnowledgeGraph(
        nodes=[
            Node(
                id="compact",
                label="/compact",
                type="Feature",
                aliases=["conversation reset"],
                chunk_ids=["alias"],
            )
        ],
        edges=[],
    )

    result = retrieve_context("conversation reset", graph, chunks)

    assert [hit.chunk.id for hit in result.chunks] == ["alias", "content-only"]
    assert result.chunks[0].score > result.chunks[1].score


def test_retrieve_context_bm25_searches_enriched_relationship_fields():
    result = retrieve_context(
        "compatibility 3 13",
        _graph(),
        _chunks(),
        top_chunks=1,
    )

    assert [hit.chunk.id for hit in result.chunks] == ["doc-chunk-0001"]
    assert [
        relationship.edge.id
        for relationship in result.chunks[0].linked_relationships
    ] == ["edge-0001"]


def test_retrieve_context_bm25_searches_chunks_without_graph_matches():
    chunks = [
        Chunk(
            id="doc-chunk-0000",
            source="doc.md",
            index=0,
            text="Deployment retries should reduce throttling errors.",
            heading_path=["Operations"],
        )
    ]

    result = retrieve_context(
        "throttling errors",
        KnowledgeGraph(nodes=[], edges=[]),
        chunks,
    )

    assert [hit.chunk.id for hit in result.chunks] == ["doc-chunk-0000"]
    assert result.chunks[0].linked_nodes == []
    assert result.chunks[0].linked_relationships == []


def test_retrieve_context_uses_semantic_chunk_search(tmp_path):
    chunks = [
        Chunk(
            id="doc-chunk-0000",
            source="doc.md",
            index=0,
            text="Setup stalls after authentication.",
            heading_path=["Deploy"],
        ),
        Chunk(
            id="doc-chunk-0001",
            source="doc.md",
            index=1,
            text="Billing exports finish normally.",
            heading_path=["Billing"],
        ),
    ]
    embedding = FakeEmbeddingClient(
        {
            "install hangs": [1.0, 0.0],
            "Setup stalls": [1.0, 0.0],
        }
    )
    store = SqliteChunkEmbeddingStore(open_database(tmp_path / "chunk_embeddings.db"))

    result = retrieve_context(
        "install hangs",
        KnowledgeGraph(nodes=[], edges=[]),
        chunks,
        embedding_client=embedding,
        chunk_embedding_store=store,
        top_chunks=1,
    )

    assert [hit.chunk.id for hit in result.chunks] == ["doc-chunk-0000"]
    assert store.exists() is True


def test_retrieve_context_reuses_and_refreshes_enriched_chunk_embedding_cache(
    tmp_path,
):
    store = SqliteChunkEmbeddingStore(open_database(tmp_path / "chunk_embeddings.db"))
    embedding = FakeEmbeddingClient(
        {
            "install hangs": [1.0, 0.0],
            "setup assistant": [1.0, 0.0],
            "deployment helper": [1.0, 0.0],
        }
    )
    chunks = [
        Chunk(
            id="doc-chunk-0000",
            source="doc.md",
            index=0,
            text="Ordinary content.",
        )
    ]
    graph = KnowledgeGraph(
        nodes=[
            Node(
                id="setup",
                label="Setup",
                type="Process",
                aliases=["setup assistant"],
                chunk_ids=["doc-chunk-0000"],
            )
        ],
        edges=[],
    )

    retrieve_context(
        "install hangs",
        graph,
        chunks,
        embedding_client=embedding,
        chunk_embedding_store=store,
    )
    assert any("setup assistant" in call for call in embedding.calls)

    embedding.calls.clear()
    retrieve_context(
        "install hangs",
        graph,
        chunks,
        embedding_client=embedding,
        chunk_embedding_store=store,
    )
    assert embedding.calls == ["install hangs"]

    embedding.calls.clear()
    changed_graph = graph.model_copy(
        update={
            "nodes": [
                graph.nodes[0].model_copy(
                    update={"aliases": ["deployment helper"]}
                )
            ]
        }
    )
    retrieve_context(
        "install hangs",
        changed_graph,
        chunks,
        embedding_client=embedding,
        chunk_embedding_store=store,
    )

    assert any("deployment helper" in call for call in embedding.calls)


def test_retrieve_context_combines_weighted_normalized_fields_and_semantic_score(
    tmp_path,
):
    embedding = FakeEmbeddingClient(
        {
            "validation": [1.0, 0.0],
        }
    )
    chunks = [
        Chunk(
            id="target",
            source="doc.md",
            index=0,
            text="Validation content.",
            heading_path=["Validation"],
        )
    ]
    graph = KnowledgeGraph(
        nodes=[
            Node(
                id="validation",
                label="Validation",
                type="Validation",
                aliases=["validation"],
                chunk_ids=["target"],
            )
        ],
        edges=[],
    )

    result = retrieve_context(
        "validation",
        graph,
        chunks,
        embedding_client=embedding,
        chunk_embedding_store=SqliteChunkEmbeddingStore(
            open_database(tmp_path / "embeddings.db")
        ),
    )
    without_semantic = retrieve_context("validation", graph, chunks)

    hit = result.chunks[0]
    assert 0.0 < hit.score < 1.0
    assert hit.relevance == hit.score
    assert hit.score > without_semantic.chunks[0].score


def test_chunk_scores_match_the_scores_reported_on_returned_hits():
    prepared = prepare_chunk_retriever(_graph(), _chunks())

    scores = prepared.chunk_scores("data validation built for python")
    result = prepared.result_for_scores(
        "data validation built for python",
        scores,
        top_chunks=len(scores),
    )

    assert scores
    assert {hit.chunk.id: hit.relevance for hit in result.chunks} == scores


def test_retrieve_context_does_not_inflate_a_weak_best_semantic_match(tmp_path):
    chunks = [
        Chunk(id="only", source="doc.md", index=0, text="Recovery notes.")
    ]
    graph = KnowledgeGraph(nodes=[], edges=[])

    def _retrieve(chunk_vector, name):
        return retrieve_context(
            "storage tuning",
            graph,
            chunks,
            embedding_client=FakeEmbeddingClient(
                {"storage tuning": [1.0, 0.0], "Recovery": chunk_vector}
            ),
            chunk_embedding_store=SqliteChunkEmbeddingStore(
                open_database(tmp_path / f"{name}.db")
            ),
        )

    weak = _retrieve([0.75, 0.6614], "weak")
    strong = _retrieve([1.0, 0.0], "strong")

    assert weak.chunks[0].score < 0.1
    assert strong.chunks[0].score > 4 * weak.chunks[0].score


def test_prominent_graph_annotations_do_not_override_stronger_evidence():
    chunks = [
        Chunk(
            id="evidence",
            source="doc.md",
            index=0,
            text="Throttling recovery reduces throttling failures during recovery.",
        ),
        Chunk(
            id="graph-heavy",
            source="doc.md",
            index=1,
            text="General operations overview.",
        ),
    ]
    nodes = [
        Node(
            id=f"node-{index:02d}",
            label=f"Operational topic {index}",
            type="Concept",
            chunk_ids=["graph-heavy"],
        )
        for index in range(10)
    ]
    graph = KnowledgeGraph(nodes=nodes, edges=[])

    result = retrieve_context("throttling recovery", graph, chunks)

    assert result.chunks[0].chunk.id == "evidence"
    assert result.chunks[0].score <= 1.0


def test_retrieve_context_attaches_only_selected_chunk_graph_annotations():
    result = retrieve_context(
        "data validation built for python",
        _graph(),
        _chunks(),
        top_chunks=1,
    )

    hit = result.chunks[0]
    assert hit.chunk.id == "doc-chunk-0001"
    assert [node.id for node in hit.linked_nodes] == ["pydantic", "python"]
    assert [
        relationship.edge.id
        for relationship in hit.linked_relationships
    ] == ["edge-0001"]
    relationship = hit.linked_relationships[0]
    assert relationship.source_node.id == "pydantic"
    assert relationship.target_node.id == "python"
    assert "fastapi" not in {node.id for node in hit.linked_nodes}


def test_retrieve_context_deduplicates_chunks_and_preserves_ranked_output():
    chunks = _chunks()

    result = retrieve_context(
        "data validation built for python",
        _graph(),
        [*chunks, chunks[1]],
        top_chunks=3,
    )

    chunk_ids = [hit.chunk.id for hit in result.chunks]
    assert chunk_ids.count("doc-chunk-0001") == 1
    assert chunk_ids[0] == "doc-chunk-0001"


def test_retrieve_context_builds_chunk_centric_context_text():
    result = retrieve_context(
        "data validation built for python",
        _graph(),
        _chunks(),
        top_chunks=1,
    )

    evidence_index = result.context_text.index(
        "Pydantic is a validation library built for Python."
    )
    entities_index = result.context_text.index("Linked entities:")
    relationships_index = result.context_text.index("Linked relationships:")
    assert evidence_index < entities_index < relationships_index
    assert "Pydantic [Library] | aliases: data parser" in result.context_text
    assert "Pydantic --built_for--> Python" in result.context_text
    assert "compatibility" in result.context_text
    assert "[doc-chunk-0001 | doc.md | Python > Pydantic]" in result.context_text


def test_retrieve_context_returns_empty_result_for_no_matches():
    result = retrieve_context("xylophone", _graph(), _chunks())

    assert result.chunks == []
    assert result.sources == []
    assert result.references == []
    assert result.context_text == "Query: xylophone\n\nEvidence:\n- None"


def test_retrieve_context_merges_pdf_page_references_and_formats_evidence():
    chunks = [
        Chunk(
            id="manual-0000",
            source="manual.pdf",
            index=0,
            text="Deployment guidance.",
            page_start=4,
            page_end=5,
        ),
        Chunk(
            id="manual-0001",
            source="manual.pdf",
            index=1,
            text="Deployment checklist.",
            page_start=6,
            page_end=6,
        ),
    ]

    result = retrieve_context(
        "deployment",
        KnowledgeGraph(nodes=[], edges=[]),
        chunks,
    )

    assert [reference.model_dump() for reference in result.references] == [
        {"source": "manual.pdf", "page_start": 4, "page_end": 6}
    ]
    assert "[manual-0000 | manual.pdf | pp. 4-5]" in result.context_text
    assert "[manual-0001 | manual.pdf | p. 6]" in result.context_text
