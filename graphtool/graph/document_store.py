import sqlite3
from pathlib import Path

from graphtool.graph.sqlite_sync import sync_source_payloads
from graphtool.graph.types import (
    Edge,
    GraphMetadata,
    KnowledgeGraph,
    Node,
)
from graphtool.storage import as_connection, transaction


class SqliteGraphStore:
    """SQLite-backed per-document graph store."""

    def __init__(
        self,
        conn_or_path: sqlite3.Connection | str | Path,
    ) -> None:
        self._conn = as_connection(conn_or_path)

    def save(self, graph: KnowledgeGraph) -> None:
        if graph.metadata is None:
            raise ValueError("Cannot save graph without metadata.source.")
        metadata = graph.metadata
        with transaction(self._conn):
            self._conn.execute(
                "INSERT INTO graph_metadata "
                "(source, content_hash, model, created_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(source) DO UPDATE SET "
                "content_hash = excluded.content_hash, "
                "model = excluded.model, "
                "created_at = excluded.created_at",
                (
                    metadata.source,
                    metadata.content_hash,
                    metadata.model,
                    metadata.created_at.isoformat(),
                ),
            )
            sync_source_payloads(
                self._conn,
                "graph_nodes",
                "node_id",
                metadata.source,
                ((node.id, node.model_dump_json()) for node in graph.nodes),
            )
            sync_source_payloads(
                self._conn,
                "graph_edges",
                "edge_id",
                metadata.source,
                ((edge.id, edge.model_dump_json()) for edge in graph.edges),
            )

    def load(self, source: str) -> KnowledgeGraph:
        metadata_row = self._conn.execute(
            "SELECT source, content_hash, model, created_at "
            "FROM graph_metadata WHERE source = ?",
            (source,),
        ).fetchone()
        if metadata_row is None:
            raise FileNotFoundError(f"Graph for {source!r} was not found.")
        nodes = [
            Node.model_validate_json(row["payload"])
            for row in self._conn.execute(
                "SELECT payload FROM graph_nodes "
                "WHERE source = ? ORDER BY rowid",
                (source,),
            )
        ]
        edges = [
            Edge.model_validate_json(row["payload"])
            for row in self._conn.execute(
                "SELECT payload FROM graph_edges "
                "WHERE source = ? ORDER BY rowid",
                (source,),
            )
        ]
        return KnowledgeGraph(
            nodes=nodes,
            edges=edges,
            metadata=_metadata_from_row(metadata_row),
        )

    def load_all(self) -> list[KnowledgeGraph]:
        sources = [metadata.source for metadata in self.load_metadata()]
        return [self.load(source) for source in sources]

    def load_metadata(self) -> list[GraphMetadata]:
        return [
            _metadata_from_row(row)
            for row in self._conn.execute(
                "SELECT source, content_hash, model, created_at "
                "FROM graph_metadata ORDER BY source"
            )
        ]

    def transaction(self):
        return transaction(self._conn)

    def exists(self, source: str) -> bool:
        row = self._conn.execute(
            "SELECT EXISTS("
            "SELECT 1 FROM graph_metadata WHERE source = ? LIMIT 1)",
            (source,),
        ).fetchone()
        return bool(row[0])

    def delete(self, source: str) -> None:
        with transaction(self._conn):
            self._conn.execute("DELETE FROM graph_nodes WHERE source = ?", (source,))
            self._conn.execute("DELETE FROM graph_edges WHERE source = ?", (source,))
            self._conn.execute(
                "DELETE FROM graph_metadata WHERE source = ?",
                (source,),
            )


def _metadata_from_row(row: sqlite3.Row) -> GraphMetadata:
    return GraphMetadata.model_validate(
        {
            "source": row["source"],
            "content_hash": row["content_hash"],
            "model": row["model"],
            "created_at": row["created_at"],
        }
    )
