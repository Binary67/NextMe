import json
from datetime import datetime, timezone
from logging import Logger
from threading import Event, Lock
from typing import TypeVar, cast

import pytest
from pydantic import ValidationError

from graphtool.chunking.types import Chunk
from graphtool.graph.combiner import combine_knowledge_graphs
from graphtool.graph.extraction import (
    chunk_messages as _chunk_messages,
    extraction_cache_key as _extraction_cache_key,
)
from graphtool.graph.extraction_store import (
    ExtractedEdge as _ExtractedEdge,
    ExtractedKnowledgeGraph as _ExtractedKnowledgeGraph,
    ExtractedNode as _ExtractedNode,
    JsonChunkExtractionStore,
)
from graphtool.graph.generator import generate_knowledge_graph
from graphtool.graph.taxonomy_stores import SqliteTaxonomySuggestionStore
from graphtool.graph.types import Edge, KnowledgeGraph, Node
from graphtool.llm.types import LLMMessage
from graphtool.run_logging import configure_run_logger
from graphtool.storage import open_database

T = TypeVar("T")


class FakeLLM:
    def __init__(
        self,
        responses: list[_ExtractedKnowledgeGraph | ValidationError],
        *,
        text_model: str = "fake-text-model",
    ) -> None:
        self.responses = responses
        self.text_model = text_model
        self.calls: list[tuple[list[LLMMessage], type]] = []

    def generate_text(self, messages):
        raise NotImplementedError

    def generate_structured(self, messages, response_model: type[T]) -> T:
        self.calls.append((list(messages), response_model))
        response = self.responses[len(self.calls) - 1]
        if isinstance(response, ValidationError):
            raise response
        return cast(T, response)


class RecordingTaxonomySuggestionStore:
    def __init__(self) -> None:
        self.calls = []

    def append_many(self, records) -> None:
        self.calls.append(list(records))


class FailingResolver:
    def combine(self, graphs):
        raise RuntimeError("resolution failed")


class CoordinatedLLM:
    text_model = "fake-text-model"

    def __init__(self, *, finish_out_of_order: bool, missing_target: bool = False):
        self.finish_out_of_order = finish_out_of_order
        self.missing_target = missing_target
        self.first_started = Event()
        self.second_completed = Event()
        self.lock = Lock()
        self.active_calls = 0
        self.max_active_calls = 0
        self.completion_order: list[str] = []

    def generate_structured(self, messages, response_model: type[T]) -> T:
        prompt = messages[-1].content
        label = "First" if "# First" in prompt else "Second"

        with self.lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)

        try:
            if self.finish_out_of_order:
                if label == "First":
                    self.first_started.set()
                    if not self.second_completed.wait(timeout=2):
                        raise TimeoutError("second chunk did not run concurrently")
                else:
                    if not self.first_started.wait(timeout=2):
                        raise TimeoutError("first chunk did not start")
                    with self.lock:
                        self.completion_order.append(label)
                    self.second_completed.set()

            edges = [
                _ExtractedEdge(
                    id=f"{label.casefold()}-edge",
                    source_ref=label.casefold(),
                    target_ref=(
                        "missing-node" if self.missing_target else label.casefold()
                    ),
                    label=f"mentions_{label.casefold()}",
                )
            ]
            graph = _extracted_graph(
                nodes=[
                    _ExtractedNode(
                        ref=label.casefold(),
                        label=label,
                        type="unclassified",
                        suggested_type="test entity",
                    )
                ],
                edges=edges,
            )
            if not self.finish_out_of_order or label == "First":
                with self.lock:
                    self.completion_order.append(label)
            return cast(T, graph)
        finally:
            with self.lock:
                self.active_calls -= 1


def _chunk(
    chunk_id: str = "doc-chunk-0000",
    index: int = 0,
    text: str = "# Python\nA dynamic language.",
    heading_path: list[str] | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        source="doc.md",
        index=index,
        text=text,
        heading_path=heading_path or ["Python"],
    )


def _extracted_graph(
    nodes: list[_ExtractedNode] | None = None,
    edges: list[_ExtractedEdge] | None = None,
) -> _ExtractedKnowledgeGraph:
    return _ExtractedKnowledgeGraph(nodes=nodes or [], edges=edges or [])


def _scoped_id(index: int, chunk_id: str = "doc-chunk-0000") -> str:
    return f"{chunk_id}::node-{index:04d}"


def test_generate_knowledge_graph_invokes_structured_generation_per_chunk():
    fake = FakeLLM([_extracted_graph(), _extracted_graph()])
    chunks = [
        _chunk("doc-chunk-0000", 0),
        _chunk("doc-chunk-0001", 1, "## Pydantic\nValidation.", ["Python", "Pydantic"]),
    ]

    generate_knowledge_graph(
        chunks, "doc.md", fake, content_hash="hash", max_workers=1
    )

    assert len(fake.calls) == 2
    messages, response_model = fake.calls[0]
    assert response_model is _ExtractedKnowledgeGraph
    assert messages[0].role == "system"
    assert "knowledge graph" in messages[0].content
    assert messages[1].role == "user"
    assert "Chunk ID:" not in messages[1].content
    assert "Source:" not in messages[1].content
    assert "Context only, do not extract this as graph content" in messages[1].content
    assert "Heading path: Python" in messages[1].content
    assert "Markdown content:" in messages[1].content
    assert "# Python" in messages[1].content


def test_generate_knowledge_graph_prompt_keeps_metadata_out_of_graph_content():
    fake = FakeLLM([_extracted_graph()])

    generate_knowledge_graph([_chunk()], "doc.md", fake, content_hash="hash")

    messages, _ = fake.calls[0]
    system_prompt = messages[0].content
    user_prompt = messages[1].content

    assert "important domain entities" in system_prompt
    assert "Do not create nodes for prompt metadata" in system_prompt
    assert "source file paths" in system_prompt
    assert "Table contents can contain useful facts" in system_prompt
    assert "unique temporary ref" in system_prompt
    assert "source_ref and target_ref" in system_prompt
    assert "Chunk ID:" not in user_prompt
    assert "Source:" not in user_prompt
    assert "Context only, do not extract this as graph content" in user_prompt
    assert "Heading path: Python" in user_prompt


def test_generate_knowledge_graph_attaches_metadata():
    fake = FakeLLM([_extracted_graph()])

    graph = generate_knowledge_graph(
        [_chunk()], "doc.md", fake, content_hash="hash"
    )

    assert graph.metadata is not None
    assert graph.metadata.source == "doc.md"
    assert graph.metadata.content_hash == "hash"
    assert graph.metadata.created_at <= datetime.now(timezone.utc)


def test_generate_knowledge_graph_attaches_chunk_ids_to_nodes_and_edges():
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[_ExtractedNode(ref="python", label="Python", type="concept")],
                edges=[
                    _ExtractedEdge(
                        id="e1",
                        source_ref="python",
                        target_ref="python",
                        label="self",
                    )
                ],
            )
        ]
    )

    graph = generate_knowledge_graph(
        [_chunk()], "doc.md", fake, content_hash="hash"
    )

    assert graph.nodes[0].id == _scoped_id(1)
    assert graph.nodes[0].chunk_ids == ["doc-chunk-0000"]
    assert graph.edges[0].id == "edge-0001"
    assert graph.edges[0].source == _scoped_id(1)
    assert graph.edges[0].target == _scoped_id(1)
    assert graph.edges[0].chunk_ids == ["doc-chunk-0000"]


def test_generate_knowledge_graph_filters_structural_nodes_and_edges():
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[
                    _ExtractedNode(ref="skills", label="Skills", type="feature"),
                    _ExtractedNode(
                        ref="workflows",
                        label="Reusable workflows",
                        type="concept",
                    ),
                    _ExtractedNode(
                        ref="chunk-wrapper",
                        label="Chunk: Python",
                        type="concept",
                    ),
                    _ExtractedNode(
                        ref="table",
                        label="Table",
                        type="unclassified",
                        suggested_type="table",
                    ),
                    _ExtractedNode(ref="source-path", label="doc.md", type="concept"),
                ],
                edges=[
                    _ExtractedEdge(
                        id="fact-edge",
                        source_ref="skills",
                        target_ref="workflows",
                        label="used_for",
                    ),
                    _ExtractedEdge(
                        id="table-edge",
                        source_ref="skills",
                        target_ref="table",
                        label="appears_in",
                    ),
                    _ExtractedEdge(
                        id="source-edge",
                        source_ref="source-path",
                        target_ref="skills",
                        label="contains",
                    ),
                ],
            )
        ]
    )

    graph = generate_knowledge_graph(
        [_chunk()], "doc.md", fake, content_hash="hash"
    )

    assert {node.id for node in graph.nodes} == {
        _scoped_id(1),
        _scoped_id(2),
    }
    assert len(graph.edges) == 1
    assert graph.edges[0].id == "edge-0001"
    assert graph.edges[0].source == _scoped_id(1)
    assert graph.edges[0].target == _scoped_id(2)
    assert graph.edges[0].label == "used_for"


def test_generate_knowledge_graph_keeps_meaningful_document_type():
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[
                    _ExtractedNode(
                        ref="plugin-guide",
                        label="Plugin guide",
                        type="document",
                    ),
                ],
            )
        ]
    )

    graph = generate_knowledge_graph(
        [_chunk()], "doc.md", fake, content_hash="hash"
    )

    assert [node.id for node in graph.nodes] == [_scoped_id(1)]
    assert graph.nodes[0].type == "document"


def test_extracted_knowledge_graph_rejects_duplicate_node_refs():
    with pytest.raises(
        ValidationError,
        match="extracted node refs must be unique: 'openai'",
    ):
        _extracted_graph(
            nodes=[
                _ExtractedNode(ref="openai", label="OpenAI", type="organization"),
                _ExtractedNode(
                    ref="openai",
                    label="OpenAI organization",
                    type="organization",
                ),
            ],
        )


def test_generate_knowledge_graph_retries_and_caches_only_valid_response(tmp_path):
    with pytest.raises(ValidationError) as error:
        _extracted_graph(
            nodes=[
                _ExtractedNode(ref="duplicate", label="OpenAI", type="organization"),
                _ExtractedNode(ref="duplicate", label="ChatGPT", type="product"),
            ]
        )
    fake = FakeLLM(
        [
            error.value,
            _extracted_graph(
                nodes=[
                    _ExtractedNode(ref="valid", label="OpenAI", type="organization")
                ]
            ),
        ]
    )
    extraction_store = JsonChunkExtractionStore(tmp_path / "chunk_extractions")

    graph = generate_knowledge_graph(
        [_chunk()],
        "doc.md",
        fake,
        content_hash="hash",
        extraction_store=extraction_store,
    )
    cached = FakeLLM([])
    cached_graph = generate_knowledge_graph(
        [_chunk()],
        "doc.md",
        cached,
        content_hash="hash",
        extraction_store=extraction_store,
    )

    assert len(fake.calls) == 2
    assert [node.id for node in graph.nodes] == [_scoped_id(1)]
    assert cached.calls == []
    assert cached_graph.nodes == graph.nodes


def test_generate_knowledge_graph_renumbers_duplicate_raw_edge_ids_within_chunk():
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[
                    _ExtractedNode(ref="python", label="Python", type="concept"),
                    _ExtractedNode(ref="pydantic", label="Pydantic", type="tool"),
                    _ExtractedNode(ref="fastapi", label="FastAPI", type="tool"),
                ],
                edges=[
                    _ExtractedEdge(
                        id="duplicate-edge",
                        source_ref="pydantic",
                        target_ref="python",
                        label="built_for",
                    ),
                    _ExtractedEdge(
                        id="duplicate-edge",
                        source_ref="fastapi",
                        target_ref="pydantic",
                        label="uses",
                    ),
                ],
            )
        ]
    )

    graph = generate_knowledge_graph(
        [_chunk()], "doc.md", fake, content_hash="hash"
    )

    assert [(edge.id, edge.source, edge.target, edge.label) for edge in graph.edges] == [
        (
            "edge-0001",
            _scoped_id(2),
            _scoped_id(1),
            "built_for",
        ),
        (
            "edge-0002",
            _scoped_id(3),
            _scoped_id(2),
            "uses",
        ),
    ]


def test_generate_knowledge_graph_deduplicates_semantic_edges_within_chunk():
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[
                    _ExtractedNode(ref="python", label="Python", type="concept"),
                    _ExtractedNode(ref="pydantic", label="Pydantic", type="tool"),
                ],
                edges=[
                    _ExtractedEdge(
                        id="first-edge",
                        source_ref="pydantic",
                        target_ref="python",
                        label="built_for",
                    ),
                    _ExtractedEdge(
                        id="second-edge",
                        source_ref="pydantic",
                        target_ref="python",
                        label="built_for",
                    ),
                ],
            )
        ]
    )

    graph = generate_knowledge_graph(
        [_chunk()], "doc.md", fake, content_hash="hash"
    )

    assert len(graph.edges) == 1
    assert graph.edges[0].id == "edge-0001"
    assert graph.edges[0].source == _scoped_id(2)
    assert graph.edges[0].target == _scoped_id(1)
    assert graph.edges[0].label == "built_for"
    assert graph.edges[0].chunk_ids == ["doc-chunk-0000"]


def test_generate_knowledge_graph_drops_and_records_edges_with_missing_nodes(tmp_path):
    dropped_edges_path = tmp_path / "dropped_edges.jsonl"
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[
                    _ExtractedNode(ref="python", label="Python", type="concept"),
                    _ExtractedNode(ref="pydantic", label="Pydantic", type="tool"),
                ],
                edges=[
                    _ExtractedEdge(
                        id="valid-edge",
                        source_ref="pydantic",
                        target_ref="python",
                        label="built_for",
                    ),
                    _ExtractedEdge(
                        id="missing-edge",
                        source_ref="pydantic",
                        target_ref="missing-node",
                        label="mentions",
                    ),
                ],
            )
        ]
    )

    graph = generate_knowledge_graph(
        [_chunk()],
        "doc.md",
        fake,
        content_hash="hash",
        dropped_edges_path=dropped_edges_path,
    )

    assert len(graph.edges) == 1
    assert graph.edges[0].id == "edge-0001"
    assert graph.edges[0].source == _scoped_id(2)
    assert graph.edges[0].target == _scoped_id(1)

    records = [
        json.loads(line)
        for line in dropped_edges_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["source"] == "doc.md"
    assert records[0]["chunk_id"] == "doc-chunk-0000"
    assert records[0]["edge_id"] == "missing-edge"
    assert records[0]["label"] == "mentions"
    assert records[0]["edge_source"] == "pydantic"
    assert records[0]["edge_target"] == "missing-node"
    assert records[0]["missing"] == ["target"]
    assert records[0]["created_at"]


def test_generate_knowledge_graph_logs_dropped_edges_to_run_log(tmp_path):
    logger = configure_run_logger(tmp_path / "logs")
    try:
        fake = FakeLLM(
            [
                _extracted_graph(
                    nodes=[
                        _ExtractedNode(ref="python", label="Python", type="concept"),
                    ],
                    edges=[
                        _ExtractedEdge(
                            id="missing-edge",
                            source_ref="python",
                            target_ref="missing-node",
                            label="mentions",
                        )
                    ],
                )
            ]
        )

        generate_knowledge_graph(
            [_chunk()], "doc.md", fake, content_hash="hash"
        )
        _flush_logger(logger)

        log_files = list((tmp_path / "logs").glob("graphtool-*.log"))
        assert len(log_files) == 1
        assert (
            "WARNING Skipped extracted edge missing-edge in doc-chunk-0000: "
            "missing target node missing-node"
        ) in log_files[0].read_text(encoding="utf-8")
    finally:
        _close_logger(logger)


def test_generate_knowledge_graph_logs_generation_counters(tmp_path):
    logger = configure_run_logger(tmp_path / "logs")
    try:
        fake = FakeLLM(
            [
                _extracted_graph(
                    nodes=[
                        _ExtractedNode(ref="skills", label="Skills", type="feature"),
                        _ExtractedNode(ref="workflow", label="Workflow", type="concept"),
                        _ExtractedNode(
                            ref="table",
                            label="Table",
                            type="unclassified",
                            suggested_type="table",
                        ),
                    ],
                    edges=[
                        _ExtractedEdge(
                            id="fact-edge",
                            source_ref="skills",
                            target_ref="workflow",
                            label="supports",
                        ),
                        _ExtractedEdge(
                            id="table-edge",
                            source_ref="skills",
                            target_ref="table",
                            label="appears_in",
                        ),
                    ],
                )
            ]
        )

        generate_knowledge_graph(
            [_chunk()], "doc.md", fake, content_hash="hash"
        )
        _flush_logger(logger)

        log_files = list((tmp_path / "logs").glob("graphtool-*.log"))
        assert len(log_files) == 1
        text = log_files[0].read_text(encoding="utf-8")
        assert (
            "DEBUG Generated chunk graph source=doc.md chunk=doc-chunk-0000 "
            "raw_nodes=3 kept_nodes=2 dropped_structural_nodes=1 "
            "raw_edges=2 kept_edges=1 dropped_edges=1"
        ) in text
        assert (
            "DEBUG Generated document graph source=doc.md chunks=1 "
            "raw_nodes=3 kept_nodes=2 dropped_structural_nodes=1 "
            "raw_edges=2 kept_edges=1 dropped_edges=1 final_nodes=2 final_edges=1 "
            "cached_chunks=0 generated_chunks=1 extraction_requests=1"
        ) in text
    finally:
        _close_logger(logger)


def test_generate_knowledge_graph_scopes_reused_node_refs_across_chunks():
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[
                    _ExtractedNode(ref="node-1", label="OpenAI", type="organization"),
                    _ExtractedNode(ref="node-2", label="ChatGPT", type="product"),
                ],
                edges=[
                    _ExtractedEdge(
                        id="first-edge",
                        source_ref="node-1",
                        target_ref="node-2",
                        label="develops",
                    )
                ],
            ),
            _extracted_graph(
                nodes=[
                    _ExtractedNode(ref="node-1", label="Claude", type="product"),
                    _ExtractedNode(ref="node-2", label="Anthropic", type="organization"),
                ],
                edges=[
                    _ExtractedEdge(
                        id="second-edge",
                        source_ref="node-2",
                        target_ref="node-1",
                        label="develops",
                    )
                ],
            ),
        ]
    )
    chunks = [
        _chunk("doc-chunk-0000", 0),
        _chunk("doc-chunk-0001", 1, "## Pydantic\nValidation.", ["Python", "Pydantic"]),
    ]

    graph = generate_knowledge_graph(
        chunks, "doc.md", fake, content_hash="hash", max_workers=1
    )

    assert {node.id: node.label for node in graph.nodes} == {
        _scoped_id(1, "doc-chunk-0000"): "OpenAI",
        _scoped_id(2, "doc-chunk-0000"): "ChatGPT",
        _scoped_id(1, "doc-chunk-0001"): "Claude",
        _scoped_id(2, "doc-chunk-0001"): "Anthropic",
    }
    assert {(edge.source, edge.target, edge.label) for edge in graph.edges} == {
        (
            _scoped_id(1, "doc-chunk-0000"),
            _scoped_id(2, "doc-chunk-0000"),
            "develops",
        ),
        (
            _scoped_id(2, "doc-chunk-0001"),
            _scoped_id(1, "doc-chunk-0001"),
            "develops",
        ),
    }


def test_extracted_knowledge_graph_schema_is_strict_for_openai():
    schema = _ExtractedKnowledgeGraph.model_json_schema()
    node_properties = _ExtractedNode.model_json_schema()["properties"]
    edge_properties = _ExtractedEdge.model_json_schema()["properties"]

    assert "ref" in node_properties
    assert "id" not in node_properties
    assert "source_ref" in edge_properties
    assert "target_ref" in edge_properties
    assert "properties" not in node_properties
    assert "properties" not in edge_properties
    assert not _has_additional_properties_true(schema)


def test_extracted_node_rejects_unknown_canonical_type():
    with pytest.raises(ValidationError):
        _ExtractedNode(
            ref="marketplace",
            label="Marketplace",
            type="distribution_channel",
        )


def test_extracted_node_requires_suggested_type_for_unclassified():
    with pytest.raises(ValidationError, match="suggested_type is required"):
        _ExtractedNode(
            ref="marketplace",
            label="Marketplace",
            type="unclassified",
        )


def test_generate_knowledge_graph_records_taxonomy_suggestions(tmp_path):
    suggestion_store = SqliteTaxonomySuggestionStore(
        open_database(tmp_path / "suggestions.db")
    )
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[
                    _ExtractedNode(
                        ref="marketplace",
                        label="Marketplace",
                        type="unclassified",
                        suggested_type="distribution channel",
                    )
                ],
                edges=[],
            )
        ]
    )

    generate_knowledge_graph(
        [_chunk()],
        "doc.md",
        fake,
        content_hash="hash",
        taxonomy_suggestion_store=suggestion_store,
    )

    records = suggestion_store.load()
    assert len(records) == 1
    assert records[0].suggested_type == "distribution channel"
    assert records[0].normalized_suggested_type == "distribution_channel"
    assert records[0].node_id == _scoped_id(1)
    assert records[0].node_label == "Marketplace"
    assert records[0].current_type == "unclassified"
    assert records[0].source == "doc.md"
    assert records[0].chunk_id == "doc-chunk-0000"
    assert records[0].created_at


def test_generate_knowledge_graph_buffers_taxonomy_suggestion_writes():
    suggestion_store = RecordingTaxonomySuggestionStore()
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[
                    _ExtractedNode(
                        ref="marketplace",
                        label="Marketplace",
                        type="unclassified",
                        suggested_type="distribution channel",
                    )
                ],
            ),
            _extracted_graph(
                nodes=[
                    _ExtractedNode(
                        ref="registry",
                        label="Registry",
                        type="unclassified",
                        suggested_type="distribution channel",
                    )
                ],
            ),
        ]
    )
    chunks = [
        _chunk("doc-chunk-0000", 0),
        _chunk("doc-chunk-0001", 1, "## Registry\nIndex.", ["Python", "Registry"]),
    ]

    generate_knowledge_graph(
        chunks,
        "doc.md",
        fake,
        content_hash="hash",
        taxonomy_suggestion_store=suggestion_store,
        max_workers=1,
    )

    assert len(suggestion_store.calls) == 1
    records = suggestion_store.calls[0]
    assert [record.node_id for record in records] == [
        _scoped_id(1, "doc-chunk-0000"),
        _scoped_id(1, "doc-chunk-0001"),
    ]
    assert [record.chunk_id for record in records] == [
        "doc-chunk-0000",
        "doc-chunk-0001",
    ]


def test_generate_knowledge_graph_processes_chunks_concurrently_in_input_order():
    fake = CoordinatedLLM(finish_out_of_order=True)
    suggestion_store = RecordingTaxonomySuggestionStore()
    chunks = [
        _chunk("doc-chunk-0000", 0, "# First\nFirst text.", ["First"]),
        _chunk("doc-chunk-0001", 1, "# Second\nSecond text.", ["Second"]),
    ]

    graph = generate_knowledge_graph(
        chunks,
        "doc.md",
        fake,
        content_hash="hash",
        taxonomy_suggestion_store=suggestion_store,
        max_workers=2,
    )

    assert fake.max_active_calls == 2
    assert fake.completion_order == ["Second", "First"]
    assert [node.label for node in graph.nodes] == ["First", "Second"]
    assert [edge.label for edge in graph.edges] == [
        "mentions_first",
        "mentions_second",
    ]
    assert [record.chunk_id for record in suggestion_store.calls[0]] == [
        "doc-chunk-0000",
        "doc-chunk-0001",
    ]


def test_generate_knowledge_graph_with_one_worker_is_sequential():
    fake = CoordinatedLLM(finish_out_of_order=False)
    chunks = [
        _chunk("doc-chunk-0000", 0, "# First\nFirst text.", ["First"]),
        _chunk("doc-chunk-0001", 1, "# Second\nSecond text.", ["Second"]),
    ]

    graph = generate_knowledge_graph(
        chunks,
        "doc.md",
        fake,
        content_hash="hash",
        max_workers=1,
    )

    assert fake.max_active_calls == 1
    assert fake.completion_order == ["First", "Second"]
    assert [node.label for node in graph.nodes] == ["First", "Second"]


@pytest.mark.parametrize("max_workers", [0, -1])
def test_generate_knowledge_graph_requires_positive_max_workers(max_workers):
    with pytest.raises(ValueError, match="max_workers must be positive"):
        generate_knowledge_graph(
            [_chunk()],
            "doc.md",
            FakeLLM([_extracted_graph()]),
            content_hash="hash",
            max_workers=max_workers,
        )


def test_generate_knowledge_graph_serializes_concurrent_dropped_edge_writes(tmp_path):
    dropped_edges_path = tmp_path / "dropped_edges.jsonl"
    fake = CoordinatedLLM(finish_out_of_order=True, missing_target=True)
    chunks = [
        _chunk("doc-chunk-0000", 0, "# First\nFirst text.", ["First"]),
        _chunk("doc-chunk-0001", 1, "# Second\nSecond text.", ["Second"]),
    ]

    generate_knowledge_graph(
        chunks,
        "doc.md",
        fake,
        content_hash="hash",
        dropped_edges_path=dropped_edges_path,
        max_workers=2,
    )

    records = [
        json.loads(line)
        for line in dropped_edges_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {record["chunk_id"] for record in records} == {
        "doc-chunk-0000",
        "doc-chunk-0001",
    }


def test_generate_knowledge_graph_reuses_unchanged_extractions_and_rescopes_nodes(
    tmp_path,
):
    extraction_store = JsonChunkExtractionStore(tmp_path / "chunk_extractions")
    first_chunks = [
        _chunk("doc-chunk-0000", 0, "# First\nFirst text.", ["First"]),
        _chunk("doc-chunk-0001", 1, "# Second\nSecond text.", ["Second"]),
    ]
    first = FakeLLM(
        [
            _extracted_graph(
                [_ExtractedNode(ref="first", label="First", type="concept")]
            ),
            _extracted_graph(
                [_ExtractedNode(ref="second", label="Second", type="concept")]
            ),
        ]
    )
    generate_knowledge_graph(
        first_chunks,
        "doc.md",
        first,
        content_hash="first-hash",
        extraction_store=extraction_store,
        max_workers=1,
    )
    changed_chunks = [
        _chunk("doc-chunk-0000", 0, "# Second\nSecond text.", ["Second"]),
        _chunk("doc-chunk-0001", 1, "# Third\nThird text.", ["Third"]),
    ]
    changed = FakeLLM(
        [
            _extracted_graph(
                [_ExtractedNode(ref="third", label="Third", type="concept")]
            )
        ]
    )

    cached_graph = generate_knowledge_graph(
        changed_chunks,
        "doc.md",
        changed,
        content_hash="changed-hash",
        extraction_store=extraction_store,
        max_workers=1,
    )
    cold_graph = generate_knowledge_graph(
        changed_chunks,
        "doc.md",
        FakeLLM(
            [
                _extracted_graph(
                    [_ExtractedNode(ref="second", label="Second", type="concept")]
                ),
                _extracted_graph(
                    [_ExtractedNode(ref="third", label="Third", type="concept")]
                ),
            ]
        ),
        content_hash="changed-hash",
        max_workers=1,
    )

    assert len(first.calls) == 2
    assert len(changed.calls) == 1
    assert [(node.id, node.label) for node in cached_graph.nodes] == [
        ("doc-chunk-0000::node-0001", "Second"),
        ("doc-chunk-0001::node-0001", "Third"),
    ]
    assert cached_graph.nodes == cold_graph.nodes
    assert cached_graph.edges == cold_graph.edges
    assert len(extraction_store.load("doc.md")) == 2


def test_generate_knowledge_graph_deduplicates_identical_cache_misses(tmp_path):
    extraction_store = JsonChunkExtractionStore(tmp_path / "chunk_extractions")
    fake = FakeLLM(
        [_extracted_graph([_ExtractedNode(ref="same", label="Same", type="concept")])]
    )
    chunks = [
        _chunk("doc-chunk-0000", 0, "# Same\nText.", ["Same"]),
        _chunk("doc-chunk-0001", 1, "# Same\nText.", ["Same"]),
    ]

    graph = generate_knowledge_graph(
        chunks,
        "doc.md",
        fake,
        content_hash="hash",
        extraction_store=extraction_store,
        max_workers=2,
    )

    assert len(fake.calls) == 1
    assert [node.id for node in graph.nodes] == [
        "doc-chunk-0000::node-0001",
        "doc-chunk-0001::node-0001",
    ]
    assert len(extraction_store.load("doc.md")) == 1


def test_extraction_cache_key_changes_with_input_prompt_schema_and_model(monkeypatch):
    chunk = _chunk()
    messages = _chunk_messages(chunk)
    original = _extraction_cache_key(messages, "model-a")

    assert (
        _extraction_cache_key(_chunk_messages(_chunk(text="Changed")), "model-a")
        != original
    )
    assert _extraction_cache_key(
        _chunk_messages(_chunk(heading_path=["Other"])),
        "model-a",
    ) != original
    assert _extraction_cache_key(messages, "model-b") != original

    monkeypatch.setattr(
        "graphtool.graph.extraction.SYSTEM_PROMPT",
        "Changed extraction instructions.",
    )
    assert _extraction_cache_key(_chunk_messages(chunk), "model-a") != original

    original_schema = _ExtractedKnowledgeGraph.model_json_schema()
    monkeypatch.setattr(
        _ExtractedKnowledgeGraph,
        "model_json_schema",
        classmethod(lambda cls: {**original_schema, "cache_test": True}),
    )
    assert _extraction_cache_key(messages, "model-a") != original


def test_generate_knowledge_graph_does_not_replace_cache_when_resolution_fails(
    tmp_path,
):
    extraction_store = JsonChunkExtractionStore(tmp_path / "chunk_extractions")
    generate_knowledge_graph(
        [_chunk()],
        "doc.md",
        FakeLLM(
            [_extracted_graph([_ExtractedNode(ref="old", label="Old", type="concept")])]
        ),
        content_hash="old-hash",
        extraction_store=extraction_store,
    )
    original = extraction_store.load("doc.md")

    with pytest.raises(RuntimeError, match="resolution failed"):
        generate_knowledge_graph(
            [_chunk(text="# New\nNew text.", heading_path=["New"])],
            "doc.md",
            FakeLLM(
                [
                    _extracted_graph(
                        [_ExtractedNode(ref="new", label="New", type="concept")]
                    )
                ]
            ),
            content_hash="new-hash",
            resolver=FailingResolver(),
            extraction_store=extraction_store,
        )

    assert extraction_store.load("doc.md") == original


def test_combine_knowledge_graphs_merges_multiple_document_graphs():
    graph = combine_knowledge_graphs(
        [
            KnowledgeGraph(
                nodes=[
                    Node(
                        id="python",
                        label="Python",
                        type="Language",
                        chunk_ids=["first-chunk-0000"],
                    )
                ],
                edges=[
                    Edge(
                        id="first-edge",
                        source="python",
                        target="python",
                        label="mentions",
                        chunk_ids=["first-chunk-0000"],
                    )
                ],
            ),
            KnowledgeGraph(
                nodes=[
                    Node(
                        id="python",
                        label="Python 3",
                        type="Version",
                        chunk_ids=["second-chunk-0000"],
                    )
                ],
                edges=[
                    Edge(
                        id="second-edge",
                        source="python",
                        target="python",
                        label="mentions",
                        chunk_ids=["second-chunk-0000"],
                    )
                ],
            ),
        ]
    )

    assert len(graph.nodes) == 1
    assert graph.nodes[0].chunk_ids == ["first-chunk-0000", "second-chunk-0000"]
    assert len(graph.edges) == 1
    assert graph.edges[0].id == "edge-0001"
    assert graph.edges[0].chunk_ids == ["first-chunk-0000", "second-chunk-0000"]


def _has_additional_properties_true(value) -> bool:
    if isinstance(value, dict):
        return any(
            (key == "additionalProperties" and child is True)
            or _has_additional_properties_true(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_has_additional_properties_true(child) for child in value)
    return False


def _flush_logger(logger: Logger) -> None:
    for handler in logger.handlers:
        handler.flush()


def _close_logger(logger: Logger) -> None:
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
