import hashlib
import logging
from datetime import datetime, timezone
from typing import TypeVar

import pytest

from graphtool.chunking.types import Chunk
from graphtool.corpus import (
    rebuild_knowledge_base,
    synchronize_documents,
)
from graphtool.corpus_stores import SqliteCorpusStores
from graphtool.graph.extraction_store import JsonChunkExtractionStore
from graphtool.graph.extraction_store import (
    ExtractedEdge as _ExtractedEdge,
    ExtractedKnowledgeGraph as _ExtractedKnowledgeGraph,
    ExtractedNode as _ExtractedNode,
)
from graphtool.graph.resolver import EntityResolutionDecision, SemanticEntityResolver
from graphtool.graph.taxonomy_types import (
    TaxonomySuggestionRecord,
)
from graphtool.graph.types import Edge, GraphMetadata, KnowledgeGraph, Node
from graphtool.llm.types import LLMMessage
from graphtool.retrieval import ChunkEmbeddingRecord
from graphtool.run_logging import LOGGER_NAME
from graphtool.source import document_content_hash, source_key
from graphtool.storage import open_database

T = TypeVar("T")


def _stores(tmp_path) -> SqliteCorpusStores:
    return SqliteCorpusStores.from_connection(
        open_database(tmp_path / "graphtool.db")
    )


class FakeLLM:
    text_model = "fake-text-model"

    def __init__(self, responses: list[_ExtractedKnowledgeGraph]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[LLMMessage], type]] = []

    def generate_text(self, messages):
        raise NotImplementedError

    def generate_structured(self, messages, response_model: type[T]) -> T:
        self.calls.append((list(messages), response_model))
        return self.responses[len(self.calls) - 1]


class FailingLLM:
    text_model = "fake-text-model"

    def generate_structured(self, messages, response_model):
        raise RuntimeError("extraction failed")


class FakeSemanticLLM:
    embedding_model = "fake-embedding-model"
    text_model = "fake-text-model"

    def __init__(
        self,
        responses: list[_ExtractedKnowledgeGraph],
        decisions: list[EntityResolutionDecision],
        vectors: dict[str, list[float]],
    ) -> None:
        self.responses = responses
        self.decisions = decisions
        self.vectors = vectors
        self.calls: list[tuple[list[LLMMessage], type]] = []
        self.embedding_calls: list[str] = []
        self.embedding_batch_calls: list[list[str]] = []
        self._response_index = 0
        self._decision_index = 0

    def generate_text(self, messages):
        raise NotImplementedError

    def generate_structured(self, messages, response_model: type[T]) -> T:
        self.calls.append((list(messages), response_model))
        if response_model is EntityResolutionDecision:
            decision = self.decisions[self._decision_index]
            self._decision_index += 1
            return decision

        response = self.responses[self._response_index]
        self._response_index += 1
        return response

    def embed_texts(self, texts) -> list[list[float]]:
        batch = list(texts)
        self.embedding_batch_calls.append(batch)
        self.embedding_calls.extend(batch)
        return [self._vector_for(text) for text in batch]

    def _vector_for(self, text: str) -> list[float]:
        label_line = text.splitlines()[0] if text else ""
        for key, vector in self.vectors.items():
            if key in label_line:
                return vector
        return [0.0, 1.0]


class FakeEmbeddingClient:
    embedding_model = "fake-embedding-model"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[str] = []

    def embed_texts(self, texts) -> list[list[float]]:
        batch = list(texts)
        self.calls.extend(batch)
        return [self._vector_for(text) for text in batch]

    def _vector_for(self, text: str) -> list[float]:
        for marker, vector in self.vectors.items():
            if marker in text:
                return vector
        return [0.0, 1.0]


def _chunk(source: str, text: str, heading: str) -> Chunk:
    return Chunk(
        id=f"{source_key(source)}-chunk-0000",
        source=source,
        index=0,
        text=text,
        heading_path=[heading],
    )


def _extracted_graph(
    nodes: list[_ExtractedNode],
    edges: list[_ExtractedEdge] | None = None,
) -> _ExtractedKnowledgeGraph:
    return _ExtractedKnowledgeGraph(nodes=nodes, edges=edges or [])


def _graph(source: str, chunk: Chunk, node_id: str, label: str) -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes=[
            Node(
                id=node_id,
                label=label,
                type="Concept",
                properties={"topic": "validation"},
                chunk_ids=[chunk.id],
            )
        ],
        edges=[],
        metadata=GraphMetadata(
            source=source,
            content_hash=document_content_hash(chunk.text),
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    )


def _relationship_graph(
    source: str,
    chunk_id: str,
    content: str = "# Existing\nGraphTool uses Azure OpenAI.",
) -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes=[
            Node(
                id="graphtool",
                label="GraphTool",
                type="Project",
                chunk_ids=[chunk_id],
            ),
            Node(
                id="azure-openai",
                label="Azure OpenAI",
                type="Service",
                chunk_ids=[chunk_id],
            ),
        ],
        edges=[
            Edge(
                id="edge-0001",
                source="graphtool",
                target="azure-openai",
                label="uses",
                chunk_ids=[chunk_id],
            )
        ],
        metadata=GraphMetadata(
            source=source,
            content_hash=document_content_hash(content),
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    )


def test_synchronize_documents_skips_unchanged_sources(tmp_path):
    stores = _stores(tmp_path)
    graph_store = stores.graphs
    chunk_store = stores.chunks
    knowledge_base_store = stores.knowledge_base
    chunk = _chunk("docs/processed.md", "# Processed\nText.", "Processed")
    graph_store.save(_graph("docs/processed.md", chunk, "processed", "Processed"))
    rebuild_knowledge_base(stores)
    fake = FakeLLM([])

    result = synchronize_documents(
        {"docs/processed.md": "# Processed\nText."},
        stores,
        fake,
    )

    assert result.unchanged_sources == ["docs/processed.md"]
    assert result.added_sources == []
    assert result.changed_sources == []
    assert result.deleted_sources == []
    assert fake.calls == []


def test_synchronize_documents_rebuilds_legacy_ingestion_fingerprint(tmp_path):
    stores = _stores(tmp_path)
    graph_store = stores.graphs
    chunk_store = stores.chunks
    knowledge_base_store = stores.knowledge_base
    source = "docs/processed.md"
    content = "# Processed\nText."
    chunk = _chunk(source, content, "Processed")
    graph = _graph(source, chunk, "processed", "Processed")
    legacy_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    graph_store.save(
        graph.model_copy(
            update={
                "metadata": graph.metadata.model_copy(
                    update={"content_hash": legacy_hash}
                )
            }
        )
    )
    chunk_store.save(source, [chunk])
    rebuild_knowledge_base(stores)
    fake = FakeLLM([_extracted_graph([])])

    result = synchronize_documents(
        {source: content},
        stores,
        fake,
    )

    assert result.changed_sources == [source]
    assert len(fake.calls) == 1
    assert graph_store.load(source).metadata.content_hash == document_content_hash(
        content
    )


def test_synchronize_documents_reuses_unchanged_chunk_extractions_and_deletes_cache(
    tmp_path,
):
    stores = _stores(tmp_path)
    graph_store = stores.graphs
    chunk_store = stores.chunks
    knowledge_base_store = stores.knowledge_base
    extraction_store = JsonChunkExtractionStore(tmp_path / "chunk_extractions")
    source = "docs/guide.md"
    first_section = f"# First\n{'alpha ' * 4100}"
    second_section = f"# Second\n{'beta ' * 4100}"
    original = f"{first_section}\n\n{second_section}"
    initial = FakeLLM(
        [
            _extracted_graph(
                [_ExtractedNode(ref="first", label="First", type="concept")]
            ),
            _extracted_graph(
                [_ExtractedNode(ref="second", label="Second", type="concept")]
            ),
        ]
    )
    synchronize_documents(
        {source: original},
        stores,
        initial,
        chunk_extraction_store=extraction_store,
        chunk_generation_workers=1,
    )
    replacement = f"{first_section}\n\n# Third\n{'gamma ' * 4100}"
    changed = FakeLLM(
        [
            _extracted_graph(
                [_ExtractedNode(ref="third", label="Third", type="concept")]
            )
        ]
    )

    result = synchronize_documents(
        {source: replacement},
        stores,
        changed,
        chunk_extraction_store=extraction_store,
        chunk_generation_workers=1,
    )

    assert result.changed_sources == [source]
    assert len(initial.calls) == 2
    assert len(changed.calls) == 1
    assert {node.label for node in graph_store.load(source).nodes} == {
        "First",
        "Third",
    }
    assert len(extraction_store.load(source)) == 2

    synchronize_documents(
        {},
        stores,
        FakeLLM([]),
        chunk_extraction_store=extraction_store,
    )

    assert extraction_store.load(source) == {}


def test_synchronize_documents_forwards_chunk_generation_worker_validation(tmp_path):
    stores = _stores(tmp_path)
    graph_store = stores.graphs
    chunk_store = stores.chunks
    knowledge_base_store = stores.knowledge_base

    with pytest.raises(
        ValueError,
        match="chunk_generation_workers must be positive",
    ):
        synchronize_documents(
            {"docs/new.md": "# New\nText."},
            stores,
            FakeLLM([]),
            chunk_generation_workers=0,
        )

    assert graph_store.exists("docs/new.md") is False
    assert knowledge_base_store.exists() is False


def test_synchronize_documents_adds_only_pending_source(
    caplog,
    monkeypatch,
    tmp_path,
):
    stores = _stores(tmp_path)
    graph_store = stores.graphs
    chunk_store = stores.chunks
    processed_chunk = _chunk("docs/processed.md", "# Processed\nText.", "Processed")
    graph_store.save(
        _graph("docs/processed.md", processed_chunk, "processed", "Processed")
    )
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[_ExtractedNode(ref="pending", label="Pending", type="concept")]
            )
        ]
    )
    logger = logging.getLogger(LOGGER_NAME)
    monkeypatch.setattr(logger, "propagate", True)
    caplog.set_level("INFO", logger=LOGGER_NAME)

    result = synchronize_documents(
        {
            "docs/processed.md": "# Processed\nText.",
            "docs/pending.md": "# Pending\nNeeds validation.",
        },
        stores,
        fake,
    )

    assert result.added_sources == ["docs/pending.md"]
    assert result.unchanged_sources == ["docs/processed.md"]
    assert (
        "Knowledge graph changes: 1 added, 0 changed, 0 removed, 1 unchanged"
        in caplog.text
    )
    assert (
        "[1/1] Building knowledge graph: docs/pending.md (1 chunk)"
        in caplog.text
    )
    assert (
        "Built knowledge graph: docs/pending.md (1 entity, 0 relationships)"
        in caplog.text
    )
    assert len(fake.calls) == 1
    assert graph_store.exists("docs/pending.md") is True
    assert chunk_store.load("docs/pending.md")


def test_synchronize_without_semantic_resolver_preserves_scoped_knowledge_base_nodes(
    tmp_path,
):
    stores = _stores(tmp_path)
    graph_store = stores.graphs
    chunk_store = stores.chunks
    knowledge_base_store = stores.knowledge_base
    existing_graph = _relationship_graph("docs/existing.md", "existing-chunk-0000")
    graph_store.save(existing_graph)
    rebuild_knowledge_base(stores)
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[
                    _ExtractedNode(ref="graphtool", label="GraphTool", type="tool"),
                    _ExtractedNode(
                        ref="azure-openai",
                        label="Azure OpenAI",
                        type="service",
                    ),
                ],
                edges=[
                    _ExtractedEdge(
                        id="llm-edge",
                        source_ref="graphtool",
                        target_ref="azure-openai",
                        label="uses",
                    )
                ],
            )
        ]
    )
    new_source = "docs/new.md"
    new_chunk_id = f"{source_key(new_source)}-chunk-0000"

    synchronize_documents(
        {
            "docs/existing.md": "# Existing\nGraphTool uses Azure OpenAI.",
            new_source: "# GraphTool\nUses Azure OpenAI.",
        },
        stores,
        fake,
    )

    graph = knowledge_base_store.load()
    assert {node.id for node in graph.nodes} == {
        "graphtool",
        "azure-openai",
        f"{new_chunk_id}::node-0001",
        f"{new_chunk_id}::node-0002",
    }
    assert {(edge.source, edge.target, edge.label) for edge in graph.edges} == {
        ("graphtool", "azure-openai", "uses"),
        (
            f"{new_chunk_id}::node-0001",
            f"{new_chunk_id}::node-0002",
            "uses",
        ),
    }


def test_synchronize_scopes_reused_node_refs_across_documents(tmp_path):
    stores = _stores(tmp_path)
    graph_store = stores.graphs
    chunk_store = stores.chunks
    knowledge_base_store = stores.knowledge_base
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[
                    _ExtractedNode(
                        ref="node-1",
                        label="Anthropic",
                        type="organization",
                    )
                ]
            ),
            _extracted_graph(
                nodes=[
                    _ExtractedNode(
                        ref="node-1",
                        label="OpenAI",
                        type="organization",
                    )
                ]
            ),
        ]
    )
    first_source = "docs/openai.md"
    second_source = "docs/anthropic.md"
    first_chunk_id = f"{source_key(first_source)}-chunk-0000"
    second_chunk_id = f"{source_key(second_source)}-chunk-0000"

    synchronize_documents(
        {
            first_source: "# OpenAI\nAI company.",
            second_source: "# Anthropic\nAI company.",
        },
        stores,
        fake,
    )

    graph = knowledge_base_store.load()
    assert {node.id: node.label for node in graph.nodes} == {
        f"{first_chunk_id}::node-0001": "OpenAI",
        f"{second_chunk_id}::node-0001": "Anthropic",
    }


def test_rebuild_knowledge_base_resolves_only_across_document_graphs(tmp_path):
    stores = _stores(tmp_path)
    graph_store = stores.graphs
    knowledge_base_store = stores.knowledge_base
    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    graph_store.save(
        KnowledgeGraph(
            nodes=[
                Node(id="alpha", label="Alpha", type="Concept"),
                Node(id="beta", label="Beta", type="Concept"),
            ],
            edges=[],
            metadata=GraphMetadata(
                source="docs/a.md",
                content_hash="hash-a",
                created_at=created_at,
            ),
        )
    )
    graph_store.save(
        KnowledgeGraph(
            nodes=[Node(id="gamma", label="Gamma", type="Concept")],
            edges=[],
            metadata=GraphMetadata(
                source="docs/b.md",
                content_hash="hash-b",
                created_at=created_at,
            ),
        )
    )
    llm = FakeSemanticLLM(
        responses=[],
        decisions=[],
        vectors={
            "Alpha": [1.0, 0.0, 0.0],
            "Beta": [0.0, 1.0, 0.0],
            "Gamma": [0.0, 0.0, 1.0],
        },
    )
    resolver = SemanticEntityResolver(
        llm,
        llm,
        min_candidate_similarity=1.1,
    )

    graph = rebuild_knowledge_base(
        stores,
        resolver=resolver,
    )

    assert {node.id for node in graph.nodes} == {"alpha", "beta", "gamma"}
    assert llm.embedding_batch_calls == [
        [
            "label: Alpha\ntype: Concept",
            "label: Beta\ntype: Concept",
            "label: Gamma\ntype: Concept",
        ]
    ]


def test_synchronize_documents_updates_cached_knowledge_base_semantically(
    tmp_path,
):
    stores = _stores(tmp_path)
    graph_store = stores.graphs
    chunk_store = stores.chunks
    knowledge_base_store = stores.knowledge_base
    graph_embedding_store = stores.graph_embeddings
    knowledge_base_embedding_store = stores.knowledge_base_embeddings
    existing_chunk_id = "existing-chunk-0000"
    existing_graph = KnowledgeGraph(
        nodes=[
            Node(
                id="openai",
                label="OpenAI",
                type="Organization",
                chunk_ids=[existing_chunk_id],
            ),
            Node(
                id="chatgpt",
                label="ChatGPT",
                type="Product",
                chunk_ids=[existing_chunk_id],
            ),
        ],
        edges=[
            Edge(
                id="edge-0001",
                source="openai",
                target="chatgpt",
                label="develops",
                chunk_ids=[existing_chunk_id],
            )
        ],
        metadata=GraphMetadata(
            source="docs/existing.md",
            content_hash=document_content_hash("# Existing\nOpenAI develops ChatGPT."),
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    )
    graph_store.save(existing_graph)
    rebuild_knowledge_base(stores)
    fake = FakeSemanticLLM(
        responses=[
            _extracted_graph(
                nodes=[
                    _ExtractedNode(
                        ref="openai-organization",
                        label="OpenAI organization",
                        type="organization",
                    ),
                    _ExtractedNode(
                        ref="chatgpt",
                        label="ChatGPT",
                        type="product",
                    ),
                ],
                edges=[
                    _ExtractedEdge(
                        id="new-edge",
                        source_ref="openai-organization",
                        target_ref="chatgpt",
                        label="develops",
                    )
                ],
            )
        ],
        decisions=[
            EntityResolutionDecision(
                decision="merge",
                target_node_id="openai",
                confidence=0.95,
                aliases_to_add=["OpenAI Inc."],
            )
        ],
        vectors={
            "OpenAI organization": [1.0, 0.0],
            "OpenAI": [1.0, 0.0],
            "ChatGPT": [0.0, 1.0],
        },
    )
    new_source = "docs/new.md"
    new_chunk_id = f"{source_key(new_source)}-chunk-0000"

    synchronize_documents(
        {
            "docs/existing.md": "# Existing\nOpenAI develops ChatGPT.",
            new_source: "# OpenAI organization\nDevelops ChatGPT.",
        },
        stores,
        fake,
    )

    graph = knowledge_base_store.load()
    assert {node.id for node in graph.nodes} == {"openai", "chatgpt"}
    openai = next(node for node in graph.nodes if node.id == "openai")
    assert openai.aliases == ["OpenAI organization", "OpenAI Inc."]
    assert openai.chunk_ids == [existing_chunk_id, new_chunk_id]
    assert openai.provenance[1].resolution_aliases == ["OpenAI Inc."]
    assert len(graph.edges) == 1
    assert graph.edges[0].source == "openai"
    assert graph.edges[0].target == "chatgpt"
    assert graph.edges[0].chunk_ids == [existing_chunk_id, new_chunk_id]
    assert graph_embedding_store.exists(new_source) is True
    assert knowledge_base_embedding_store.exists() is True


def test_synchronize_documents_reuses_document_embedding_for_knowledge_base(
    tmp_path,
):
    stores = _stores(tmp_path)
    graph_store = stores.graphs
    chunk_store = stores.chunks
    knowledge_base_store = stores.knowledge_base
    graph_embedding_store = stores.graph_embeddings
    knowledge_base_embedding_store = stores.knowledge_base_embeddings
    fake = FakeSemanticLLM(
        responses=[
            _extracted_graph(
                nodes=[_ExtractedNode(ref="alpha", label="Alpha", type="concept")]
            )
        ],
        decisions=[],
        vectors={"Alpha": [1.0, 0.0]},
    )

    synchronize_documents(
        {"docs/alpha.md": "# Alpha\nAlpha."},
        stores,
        fake,
    )

    assert fake.embedding_calls == ["label: Alpha\ntype: concept"]


def test_synchronize_documents_uses_min_candidate_similarity_for_resolvers(
    tmp_path,
):
    stores = _stores(tmp_path)
    graph_store = stores.graphs
    chunk_store = stores.chunks
    knowledge_base_store = stores.knowledge_base
    graph_embedding_store = stores.graph_embeddings
    knowledge_base_embedding_store = stores.knowledge_base_embeddings
    existing_graph = KnowledgeGraph(
        nodes=[
            Node(
                id="openai",
                label="OpenAI",
                type="Organization",
                chunk_ids=["existing-chunk-0000"],
            )
        ],
        edges=[],
        metadata=GraphMetadata(
            source="docs/existing.md",
            content_hash=document_content_hash("# Existing\nOpenAI."),
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    )
    graph_store.save(existing_graph)
    rebuild_knowledge_base(stores)
    fake = FakeSemanticLLM(
        responses=[
            _extracted_graph(
                nodes=[
                    _ExtractedNode(
                        ref="openai-organization",
                        label="OpenAI organization",
                        type="organization",
                    )
                ]
            ),
            _extracted_graph(
                nodes=[
                    _ExtractedNode(
                        ref="openai-company",
                        label="OpenAI company",
                        type="organization",
                    )
                ]
            ),
        ],
        decisions=[],
        vectors={
            "OpenAI organization": [0.9, 0.1],
            "OpenAI company": [0.8, 0.2],
            "OpenAI": [1.0, 0.0],
        },
    )
    new_source = "docs/new.md"
    new_markdown = (
        "# OpenAI organization\nFirst mention.\n\n"
        f"{'Context. ' * 2200}\n\n"
        "# OpenAI company\nSecond mention."
    )

    synchronize_documents(
        {
            "docs/existing.md": "# Existing\nOpenAI.",
            new_source: new_markdown,
        },
        stores,
        fake,
        min_candidate_similarity=1.0,
        chunk_generation_workers=1,
    )

    document_graph = graph_store.load(new_source)
    knowledge_base = knowledge_base_store.load()
    first_chunk_id = f"{source_key(new_source)}-chunk-0000"
    second_chunk_id = f"{source_key(new_source)}-chunk-0001"

    assert {node.id for node in document_graph.nodes} == {
        f"{first_chunk_id}::node-0001",
        f"{second_chunk_id}::node-0001",
    }
    assert {node.id for node in knowledge_base.nodes} == {
        "openai",
        f"{first_chunk_id}::node-0001",
        f"{second_chunk_id}::node-0001",
    }
    assert [
        response_model
        for _, response_model in fake.calls
        if response_model is EntityResolutionDecision
    ] == []


def test_synchronize_documents_rebuilds_missing_knowledge_base_cache(tmp_path):
    stores = _stores(tmp_path)
    graph_store = stores.graphs
    chunk_store = stores.chunks
    knowledge_base_store = stores.knowledge_base
    processed_chunk = _chunk("docs/processed.md", "# Processed\nText.", "Processed")
    graph_store.save(
        _graph("docs/processed.md", processed_chunk, "processed", "Processed")
    )
    fake = FakeLLM(
        [
            _extracted_graph(
                nodes=[_ExtractedNode(ref="pending", label="Pending", type="concept")]
            )
        ]
    )

    synchronize_documents(
        {
            "docs/processed.md": "# Processed\nText.",
            "docs/pending.md": "# Pending\nNeeds validation.",
        },
        stores,
        fake,
    )

    graph = knowledge_base_store.load()
    pending_chunk_id = f"{source_key('docs/pending.md')}-chunk-0000"
    assert {node.id for node in graph.nodes} == {
        "processed",
        f"{pending_chunk_id}::node-0001",
    }


def test_synchronize_changed_document_replaces_only_its_contributions_and_caches(
    tmp_path,
):
    stores = _stores(tmp_path)
    graph_store = stores.graphs
    chunk_store = stores.chunks
    knowledge_base_store = stores.knowledge_base
    graph_embedding_store = stores.graph_embeddings
    knowledge_base_embedding_store = stores.knowledge_base_embeddings
    chunk_embedding_store = stores.chunk_embeddings
    taxonomy_store = stores.taxonomy_suggestions
    source_a = "docs/a.md"
    source_b = "docs/b.md"
    original_a = "# A\nShared has Old A."
    content_b = "# B\nShared has B."
    initial = FakeSemanticLLM(
        responses=[
            _extracted_graph(
                nodes=[
                    _ExtractedNode(ref="shared", label="Shared", type="concept"),
                    _ExtractedNode(ref="old-a", label="Old A", type="feature"),
                ],
                edges=[
                    _ExtractedEdge(
                        id="a-edge",
                        source_ref="shared",
                        target_ref="old-a",
                        label="has",
                    )
                ],
            ),
            _extracted_graph(
                nodes=[
                    _ExtractedNode(ref="shared", label="Shared", type="concept"),
                    _ExtractedNode(ref="b", label="B", type="tool"),
                ],
                edges=[
                    _ExtractedEdge(
                        id="b-edge",
                        source_ref="shared",
                        target_ref="b",
                        label="has",
                    )
                ],
            ),
        ],
        decisions=[],
        vectors={},
    )
    synchronize_documents(
        {source_a: original_a, source_b: content_b},
        stores,
        initial,
    )
    old_a_chunk_id = f"{source_key(source_a)}-chunk-0000"
    b_chunk_id = f"{source_key(source_b)}-chunk-0000"
    chunk_embedding_store.upsert(
        {
            old_a_chunk_id: ChunkEmbeddingRecord(
                chunk_id=old_a_chunk_id,
                embedding_model="model",
                embedding_input_hash="old-a",
                vector=[1.0],
            ),
            b_chunk_id: ChunkEmbeddingRecord(
                chunk_id=b_chunk_id,
                embedding_model="model",
                embedding_input_hash="b",
                vector=[1.0],
            ),
        }
    )
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    taxonomy_store.save(
        [
            TaxonomySuggestionRecord(
                suggested_type="old-a",
                normalized_suggested_type="old a",
                node_id="old-a",
                node_label="Old A",
                current_type="unclassified",
                source=source_a,
                chunk_id=old_a_chunk_id,
                created_at=timestamp,
            ),
            TaxonomySuggestionRecord(
                suggested_type="b",
                normalized_suggested_type="b",
                node_id="b",
                node_label="B",
                current_type="unclassified",
                source=source_b,
                chunk_id=b_chunk_id,
                created_at=timestamp,
            ),
        ]
    )
    original_graph = knowledge_base_store.load()
    shared_id = next(node.id for node in original_graph.nodes if node.label == "Shared")
    replacement_a = "# A\nShared has New A."
    changed = FakeSemanticLLM(
        responses=[
            _extracted_graph(
                nodes=[
                    _ExtractedNode(ref="shared", label="Shared", type="concept"),
                    _ExtractedNode(
                        ref="new-a",
                        label="New A",
                        type="capability",
                    ),
                ],
                edges=[
                    _ExtractedEdge(
                        id="new-a-edge",
                        source_ref="shared",
                        target_ref="new-a",
                        label="has",
                    )
                ],
            )
        ],
        decisions=[],
        vectors={},
    )

    result = synchronize_documents(
        {source_a: replacement_a, source_b: content_b},
        stores,
        changed,
    )

    graph = knowledge_base_store.load()
    labels = {node.label for node in graph.nodes}
    shared = next(node for node in graph.nodes if node.label == "Shared")
    assert result.changed_sources == [source_a]
    assert result.unchanged_sources == [source_b]
    assert len(changed.responses) == 1
    assert shared.id == shared_id
    assert [item.source for item in shared.provenance] == [source_b, source_a]
    assert "Old A" not in labels
    assert {"Shared", "New A", "B"} <= labels
    assert graph_store.load(source_a).metadata.content_hash == document_content_hash(
        replacement_a
    )
    assert set(chunk_embedding_store.load()) == {b_chunk_id}
    assert {record.source for record in taxonomy_store.load()} == {source_b}


def test_changed_local_node_id_does_not_overwrite_surviving_shared_entity(tmp_path):
    stores = _stores(tmp_path)
    graph_store = stores.graphs
    chunk_store = stores.chunks
    knowledge_base_store = stores.knowledge_base
    source_a = "docs/a.md"
    source_b = "docs/b.md"
    content_b = "# B\nShared."
    initial = FakeSemanticLLM(
        responses=[
            _extracted_graph(
                [_ExtractedNode(ref="shared", label="Shared", type="concept")]
            ),
            _extracted_graph(
                [_ExtractedNode(ref="shared", label="Shared", type="concept")]
            ),
        ],
        decisions=[],
        vectors={},
    )
    synchronize_documents(
        {source_a: "# A\nShared.", source_b: content_b},
        stores,
        initial,
    )
    shared_id = knowledge_base_store.load().nodes[0].id
    changed = FakeSemanticLLM(
        responses=[
            _extracted_graph(
                [
                    _ExtractedNode(
                        ref="different",
                        label="Different",
                        type="capability",
                    )
                ]
            )
        ],
        decisions=[],
        vectors={},
    )

    synchronize_documents(
        {source_a: "# A\nDifferent.", source_b: content_b},
        stores,
        changed,
    )

    graph = knowledge_base_store.load()
    assert {node.label for node in graph.nodes} == {"Shared", "Different"}
    assert next(node.id for node in graph.nodes if node.label == "Shared") == shared_id
    assert len({node.id for node in graph.nodes}) == 2


def test_synchronize_deleted_document_removes_artifacts_but_keeps_shared_nodes(
    tmp_path,
):
    stores = _stores(tmp_path)
    graph_store = stores.graphs
    chunk_store = stores.chunks
    knowledge_base_store = stores.knowledge_base
    graph_embedding_store = stores.graph_embeddings
    knowledge_base_embedding_store = stores.knowledge_base_embeddings
    source_a = "docs/a.md"
    source_b = "docs/b.md"
    initial = FakeSemanticLLM(
        responses=[
            _extracted_graph(
                nodes=[
                    _ExtractedNode(ref="shared", label="Shared", type="concept"),
                    _ExtractedNode(ref="a", label="A", type="feature"),
                ]
            ),
            _extracted_graph(
                nodes=[
                    _ExtractedNode(ref="shared", label="Shared", type="concept"),
                    _ExtractedNode(ref="b", label="B", type="tool"),
                ]
            ),
        ],
        decisions=[],
        vectors={},
    )
    synchronize_documents(
        {source_a: "# A\nShared and A.", source_b: "# B\nShared and B."},
        stores,
        initial,
    )
    shared_id = next(
        node.id
        for node in knowledge_base_store.load().nodes
        if node.label == "Shared"
    )
    deleting = FakeSemanticLLM(responses=[], decisions=[], vectors={})

    result = synchronize_documents(
        {source_b: "# B\nShared and B."},
        stores,
        deleting,
    )

    graph = knowledge_base_store.load()
    shared = next(node for node in graph.nodes if node.label == "Shared")
    assert result.deleted_sources == [source_a]
    assert graph_store.exists(source_a) is False
    assert chunk_store.load(source_a) == []
    assert graph_embedding_store.exists(source_a) is False
    assert shared.id == shared_id
    assert [item.source for item in shared.provenance] == [source_b]
    assert {node.label for node in graph.nodes} == {"Shared", "B"}
    assert set(knowledge_base_embedding_store.load()) == {
        node.id for node in graph.nodes
    }


def test_synchronize_rename_is_delete_plus_add(tmp_path):
    stores = _stores(tmp_path)
    graph_store = stores.graphs
    chunk_store = stores.chunks
    knowledge_base_store = stores.knowledge_base
    old_source = "docs/old.md"
    new_source = "docs/new.md"
    content = "# Document\nContent."
    synchronize_documents(
        {old_source: content},
        stores,
        FakeLLM(
            [
                _extracted_graph(
                    [_ExtractedNode(ref="doc", label="Doc", type="document")]
                )
            ]
        ),
    )

    result = synchronize_documents(
        {new_source: content},
        stores,
        FakeLLM(
            [
                _extracted_graph(
                    [_ExtractedNode(ref="doc", label="Doc", type="document")]
                )
            ]
        ),
    )

    assert result.added_sources == [new_source]
    assert result.deleted_sources == [old_source]
    assert graph_store.exists(old_source) is False
    assert graph_store.exists(new_source) is True


def test_failed_changed_document_extraction_preserves_active_data(tmp_path):
    stores = _stores(tmp_path)
    graph_store = stores.graphs
    chunk_store = stores.chunks
    knowledge_base_store = stores.knowledge_base
    extraction_store = JsonChunkExtractionStore(tmp_path / "chunk_extractions")
    source = "docs/a.md"
    original = "# A\nOriginal."
    synchronize_documents(
        {source: original},
        stores,
        FakeLLM(
            [_extracted_graph([_ExtractedNode(ref="a", label="A", type="concept")])]
        ),
        chunk_extraction_store=extraction_store,
    )
    original_graph = graph_store.load(source)
    original_knowledge_base = knowledge_base_store.load()
    original_extractions = extraction_store.load(source)

    with pytest.raises(RuntimeError, match="extraction failed"):
        synchronize_documents(
            {source: "# A\nReplacement."},
            stores,
            FailingLLM(),
            chunk_extraction_store=extraction_store,
        )

    assert graph_store.load(source) == original_graph
    assert knowledge_base_store.load() == original_knowledge_base
    assert chunk_store.load(source)[0].text == original
    assert extraction_store.load(source) == original_extractions


def test_failed_final_write_rolls_back_all_sqlite_stores(monkeypatch, tmp_path):
    stores = _stores(tmp_path)
    source = "docs/a.md"
    initial = FakeSemanticLLM(
        responses=[
            _extracted_graph(
                [_ExtractedNode(ref="a", label="Alpha", type="concept")]
            )
        ],
        decisions=[],
        vectors={"Alpha": [1.0, 0.0]},
    )
    synchronize_documents(
        {source: "# Alpha\nOriginal."},
        stores,
        initial,
    )
    original_graph = stores.graphs.load(source)
    original_knowledge_base = stores.knowledge_base.load()
    original_chunks = stores.chunks.load(source)
    original_graph_embeddings = stores.graph_embeddings.load(source)
    original_knowledge_embeddings = stores.knowledge_base_embeddings.load()
    changed = FakeSemanticLLM(
        responses=[
            _extracted_graph(
                [_ExtractedNode(ref="b", label="Beta", type="concept")]
            )
        ],
        decisions=[],
        vectors={"Beta": [0.0, 1.0]},
    )

    def fail_apply_delta(_delta):
        raise RuntimeError("write failed")

    monkeypatch.setattr(
        stores.knowledge_base,
        "apply_delta",
        fail_apply_delta,
    )

    with pytest.raises(RuntimeError, match="write failed"):
        synchronize_documents(
            {source: "# Beta\nReplacement."},
            stores,
            changed,
        )

    assert stores.graphs.load(source) == original_graph
    assert stores.knowledge_base.load() == original_knowledge_base
    assert stores.chunks.load(source) == original_chunks
    assert stores.graph_embeddings.load(source) == original_graph_embeddings
    assert (
        stores.knowledge_base_embeddings.load()
        == original_knowledge_embeddings
    )
