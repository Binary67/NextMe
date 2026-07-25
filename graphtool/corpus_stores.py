import sqlite3
from dataclasses import dataclass

from graphtool.chunking.store import SqliteChunkStore
from graphtool.graph.embedding_store import (
    SqliteEmbeddingStore,
    SqliteGraphEmbeddingStore,
)
from graphtool.graph.document_store import SqliteGraphStore
from graphtool.graph.knowledge_base_store import SqliteKnowledgeBaseStore
from graphtool.graph.taxonomy_stores import SqliteTaxonomySuggestionStore
from graphtool.retrieval.embedding_store import SqliteChunkEmbeddingStore
from graphtool.storage import transaction


@dataclass(frozen=True)
class SqliteCorpusStores:
    """SQLite stores participating in one corpus synchronization transaction."""

    connection: sqlite3.Connection
    graphs: SqliteGraphStore
    knowledge_base: SqliteKnowledgeBaseStore
    graph_embeddings: SqliteGraphEmbeddingStore
    knowledge_base_embeddings: SqliteEmbeddingStore
    taxonomy_suggestions: SqliteTaxonomySuggestionStore
    chunks: SqliteChunkStore
    chunk_embeddings: SqliteChunkEmbeddingStore

    @classmethod
    def from_connection(cls, connection: sqlite3.Connection) -> "SqliteCorpusStores":
        return cls(
            connection=connection,
            graphs=SqliteGraphStore(connection),
            knowledge_base=SqliteKnowledgeBaseStore(connection),
            graph_embeddings=SqliteGraphEmbeddingStore(connection),
            knowledge_base_embeddings=SqliteEmbeddingStore(connection),
            taxonomy_suggestions=SqliteTaxonomySuggestionStore(connection),
            chunks=SqliteChunkStore(connection),
            chunk_embeddings=SqliteChunkEmbeddingStore(connection),
        )

    def transaction(self):
        return transaction(self.connection)
