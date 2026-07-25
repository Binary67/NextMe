from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from graphtool.chunking import SqliteChunkStore
from graphtool.chunking.types import Chunk
from graphtool.corpus_stores import SqliteCorpusStores
from graphtool.graph import (
    JsonChunkExtractionStore,
    KnowledgeGraph,
    NodeEmbeddingRecord,
    SqliteEmbeddingStore,
    SqliteGraphStore,
    SqliteGraphEmbeddingStore,
    SqliteKnowledgeBaseStore,
    SqliteTaxonomySuggestionStore,
    filter_knowledge_graph_by_sources,
)
from graphtool.knowledge_scopes import load_knowledge_scopes, source_is_in_scope
from graphtool.llm import AzureOpenAIAudioTranscriber, AzureOpenAIClient
from graphtool.llm.config import AzureOpenAIConfig
from graphtool.retrieval import (
    ChunkEmbeddingRecord,
    DocumentSearchResult,
    PreparedDocumentRetriever,
    RetrievalResult,
    SqliteChunkEmbeddingStore,
    prepare_document_retriever,
)
from graphtool.retrieval.hybrid_retriever import (
    PreparedHybridRetriever,
    prepare_hybrid_retriever,
)
from graphtool.storage import open_database

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAX_LOG_FILES = 3


@dataclass(frozen=True)
class GraphToolPaths:
    root: Path
    documents_dir: Path
    knowledge_scopes_path: Path
    audio_transcriptions_dir: Path
    audio_transcription_glossary_path: Path
    pdf_conversions_dir: Path
    presentation_conversions_dir: Path
    chunk_extractions_dir: Path
    db_path: Path
    dropped_edges_path: Path
    logs_dir: Path
    visualizations_dir: Path


@dataclass
class GraphToolRuntime:
    paths: GraphToolPaths
    corpus_stores: SqliteCorpusStores
    chunk_extraction_store: JsonChunkExtractionStore
    fast_llm: AzureOpenAIClient
    audio_transcriber: AzureOpenAIAudioTranscriber
    knowledge_scopes: dict[str, str] = field(default_factory=dict)
    _search_retrievers: dict[str | None, PreparedHybridRetriever] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _search_graph: KnowledgeGraph | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _search_chunks_by_source: dict[str, list[Chunk]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _document_retriever: PreparedDocumentRetriever | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _search_chunk_embeddings: dict[str, ChunkEmbeddingRecord] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _search_node_embeddings: dict[str, NodeEmbeddingRecord] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def search(
        self,
        query: str,
        *,
        scope: str | None = None,
        sources: list[str] | None = None,
        top_chunks: int = 5,
    ) -> RetrievalResult:
        normalized_scope = self._validate_search_scope(scope)
        retriever = (
            self._prepare_source_retriever(sources, normalized_scope)
            if sources is not None
            else self._retriever_for_scope(normalized_scope)
        )
        return retriever.retrieve(query, top_chunks=top_chunks)

    def find_documents(
        self,
        query: str,
        *,
        scope: str | None = None,
        top_documents: int = 5,
    ) -> DocumentSearchResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Document search query must not be empty.")
        if top_documents < 1:
            raise ValueError("top_documents must be positive")
        normalized_scope = self._validate_search_scope(scope)
        assert self._document_retriever is not None
        allowed_sources = (
            None
            if normalized_scope is None
            else self._sources_for_scope(normalized_scope)
        )
        retriever = self._retriever_for_scope(normalized_scope)
        query_vector = (
            self.fast_llm.embed_texts([normalized_query])[0]
            if retriever.direct.chunk_vectors
            else None
        )
        passage_result = retriever.direct.retrieve(
            normalized_query,
            top_chunks=max(top_documents * 4, 20),
            query_vector=query_vector,
        )
        return self._document_retriever.retrieve(
            normalized_query,
            passage_hits=passage_result.chunks,
            allowed_sources=allowed_sources,
            top_documents=top_documents,
        )

    @property
    def graph_store(self) -> SqliteGraphStore:
        return self.corpus_stores.graphs

    @property
    def knowledge_base_store(self) -> SqliteKnowledgeBaseStore:
        return self.corpus_stores.knowledge_base

    @property
    def graph_embedding_store(self) -> SqliteGraphEmbeddingStore:
        return self.corpus_stores.graph_embeddings

    @property
    def knowledge_base_embedding_store(self) -> SqliteEmbeddingStore:
        return self.corpus_stores.knowledge_base_embeddings

    @property
    def taxonomy_suggestion_store(self) -> SqliteTaxonomySuggestionStore:
        return self.corpus_stores.taxonomy_suggestions

    @property
    def chunk_store(self) -> SqliteChunkStore:
        return self.corpus_stores.chunks

    @property
    def chunk_embedding_store(self) -> SqliteChunkEmbeddingStore:
        return self.corpus_stores.chunk_embeddings

    def prepare_search(self) -> None:
        graph, chunks = self._search_inputs()
        self._search_graph = graph
        self._search_chunks_by_source = {}
        for chunk in chunks:
            self._search_chunks_by_source.setdefault(chunk.source, []).append(chunk)
        self._document_retriever = prepare_document_retriever(chunks)
        self._search_retrievers = {
            None: prepare_hybrid_retriever(
                graph,
                chunks,
                embedding_client=self.fast_llm,
                chunk_embedding_store=self.chunk_embedding_store,
                node_embedding_store=self.knowledge_base_embedding_store,
            )
        }
        self._search_chunk_embeddings = self.chunk_embedding_store.load()
        self._search_node_embeddings = self.knowledge_base_embedding_store.load()

    def _validate_search_scope(self, scope: str | None) -> str | None:
        if self._search_graph is None:
            raise RuntimeError(
                "Search is not prepared. Call prepare_search after synchronization."
            )
        normalized_scope = scope.strip().casefold() if scope is not None else None
        if normalized_scope is not None and normalized_scope not in self.knowledge_scopes:
            available = ", ".join(self.knowledge_scopes) or "none"
            raise ValueError(
                f"Unknown knowledge scope {scope!r}. Available scopes: {available}."
            )
        return normalized_scope

    def _retriever_for_scope(
        self,
        scope: str | None,
    ) -> PreparedHybridRetriever:
        retriever = self._search_retrievers.get(scope)
        if retriever is None:
            assert scope is not None
            retriever = self._prepare_scoped_retriever(scope)
            self._search_retrievers[scope] = retriever
        return retriever

    def _prepare_source_retriever(
        self,
        sources: list[str],
        scope: str | None,
    ) -> PreparedHybridRetriever:
        normalized_sources = list(dict.fromkeys(source.strip() for source in sources))
        if not normalized_sources or any(not source for source in normalized_sources):
            raise ValueError("At least one non-empty source must be provided.")
        unknown_sources = [
            source
            for source in normalized_sources
            if source not in self._search_chunks_by_source
        ]
        if unknown_sources:
            joined = ", ".join(repr(source) for source in unknown_sources)
            raise ValueError(f"Unknown document source: {joined}.")
        if scope is not None:
            scoped_sources = self._sources_for_scope(scope)
            outside_scope = [
                source
                for source in normalized_sources
                if source not in scoped_sources
            ]
            if outside_scope:
                joined = ", ".join(repr(source) for source in outside_scope)
                raise ValueError(
                    f"Document sources are outside knowledge scope {scope!r}: "
                    f"{joined}."
                )

        chunks = self._chunks_for_sources(normalized_sources)
        assert self._search_graph is not None
        graph = filter_knowledge_graph_by_sources(
            self._search_graph,
            normalized_sources,
        )
        return prepare_hybrid_retriever(
            graph,
            chunks,
            embedding_client=self.fast_llm,
            chunk_embedding_store=_MemoryChunkEmbeddingStore(
                self._search_chunk_embeddings
            ),
            node_embedding_store=_MemoryNodeEmbeddingStore(
                self._search_node_embeddings
            ),
        )

    def _prepare_scoped_retriever(
        self,
        scope: str,
    ) -> PreparedHybridRetriever:
        assert self._search_graph is not None
        sources = self._sources_for_scope(scope)
        chunks = self._chunks_for_sources(sorted(sources))
        graph = filter_knowledge_graph_by_sources(self._search_graph, sources)
        return prepare_hybrid_retriever(
            graph,
            chunks,
            embedding_client=self.fast_llm,
            chunk_embedding_store=_MemoryChunkEmbeddingStore(
                self._search_chunk_embeddings
            ),
            node_embedding_store=None,
        )

    def _sources_for_scope(self, scope: str) -> set[str]:
        prefix = self.knowledge_scopes[scope]
        return {
            source
            for source in self._search_chunks_by_source
            if source_is_in_scope(source, prefix)
        }

    def _chunks_for_sources(self, sources: list[str]) -> list[Chunk]:
        return [
            chunk
            for source in sources
            for chunk in self._search_chunks_by_source[source]
        ]

    def _search_inputs(self) -> tuple[KnowledgeGraph, list[Chunk]]:
        if not self.knowledge_base_store.exists():
            raise FileNotFoundError(
                "Knowledge base not found. Synchronize documents before searching."
            )
        return self.knowledge_base_store.load(), self.chunk_store.load_all()


def default_paths(root: str | Path | None = None) -> GraphToolPaths:
    project_root = Path(root) if root is not None else DEFAULT_PROJECT_ROOT
    data_dir = project_root / "data"
    return GraphToolPaths(
        root=project_root,
        documents_dir=project_root / "documents",
        knowledge_scopes_path=project_root / "config" / "knowledge_scopes.json",
        audio_transcriptions_dir=data_dir / "audio_transcriptions",
        audio_transcription_glossary_path=(
            project_root / "config" / "transcription_glossary.json"
        ),
        pdf_conversions_dir=data_dir / "pdf_conversions",
        presentation_conversions_dir=data_dir / "presentation_conversions",
        chunk_extractions_dir=data_dir / "chunk_extractions",
        db_path=data_dir / "graphtool.db",
        dropped_edges_path=data_dir / "dropped_edges.jsonl",
        logs_dir=project_root / "logs",
        visualizations_dir=data_dir / "visualizations",
    )


def create_runtime(
    config: AzureOpenAIConfig,
    *,
    paths: GraphToolPaths | None = None,
) -> GraphToolRuntime:
    runtime_paths = paths or default_paths()
    conn = open_database(runtime_paths.db_path)
    return GraphToolRuntime(
        paths=runtime_paths,
        corpus_stores=SqliteCorpusStores.from_connection(conn),
        chunk_extraction_store=JsonChunkExtractionStore(
            runtime_paths.chunk_extractions_dir
        ),
        fast_llm=AzureOpenAIClient(
            config,
            text_deployment=config.fast_deployment,
        ),
        audio_transcriber=AzureOpenAIAudioTranscriber(config),
        knowledge_scopes=load_knowledge_scopes(
            runtime_paths.knowledge_scopes_path
        ),
    )


class _MemoryChunkEmbeddingStore:
    def __init__(
        self,
        records: Mapping[str, ChunkEmbeddingRecord],
    ) -> None:
        self._records = dict(records)

    def load(self) -> dict[str, ChunkEmbeddingRecord]:
        return dict(self._records)

    def upsert(self, records: Mapping[str, ChunkEmbeddingRecord]) -> None:
        self._records.update(records)

    def delete(self, chunk_ids: list[str]) -> None:
        for chunk_id in chunk_ids:
            self._records.pop(chunk_id, None)


class _MemoryNodeEmbeddingStore:
    def __init__(
        self,
        records: Mapping[str, NodeEmbeddingRecord],
    ) -> None:
        self._records = dict(records)

    def load(self) -> dict[str, NodeEmbeddingRecord]:
        return dict(self._records)
