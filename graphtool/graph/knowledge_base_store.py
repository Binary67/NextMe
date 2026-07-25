import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from graphtool.graph.entity_matching import normalized_entity_name
from graphtool.graph.knowledge_base_rows import (
    load_edge_provenance,
    load_node_provenance,
    sync_aliases,
    sync_edges,
)
from graphtool.graph.provenance import materialize_edge, materialize_node
from graphtool.graph.sqlite_sync import sync_keyed_payloads, sync_provenance
from graphtool.graph.types import Edge, KnowledgeGraph, Node
from graphtool.storage import as_connection, transaction

_BATCH_SIZE = 400


@dataclass(frozen=True)
class KnowledgeBaseDelta:
    upserted_nodes: list[Node]
    deleted_node_ids: set[str]
    upserted_edges: list[Edge]
    deleted_edge_ids: set[str]


class SqliteKnowledgeBaseStore:
    """Incremental SQLite-backed canonical knowledge graph store."""

    def __init__(
        self,
        conn_or_path: sqlite3.Connection | str | Path,
    ) -> None:
        self._conn = as_connection(conn_or_path)

    def replace_all(self, graph: KnowledgeGraph) -> None:
        node_rows = {
            node.id: node.model_copy(update={"provenance": []}).model_dump_json()
            for node in graph.nodes
        }
        alias_rows = {
            (node.id, normalized): alias
            for node in graph.nodes
            for alias in [node.label, *node.aliases]
            if (normalized := normalized_entity_name(alias))
        }
        node_provenance_rows = {
            (
                node.id,
                provenance.source,
                provenance.content_hash,
                provenance.node_id,
            ): provenance.model_dump_json()
            for node in graph.nodes
            for provenance in node.provenance
        }
        edge_rows = {
            edge.id: (
                edge.source,
                edge.target,
                edge.model_copy(update={"provenance": []}).model_dump_json(),
            )
            for edge in graph.edges
        }
        edge_provenance_rows = {
            (
                edge.id,
                provenance.source,
                provenance.content_hash,
                provenance.edge_id,
            ): provenance.model_dump_json()
            for edge in graph.edges
            for provenance in edge.provenance
        }

        with transaction(self._conn):
            self._conn.execute(
                "INSERT OR IGNORE INTO knowledge_base_state(singleton) VALUES (1)"
            )
            sync_keyed_payloads(
                self._conn,
                "knowledge_base_nodes",
                "node_id",
                node_rows,
            )
            sync_aliases(self._conn, alias_rows)
            sync_provenance(
                self._conn,
                "knowledge_base_node_provenance",
                (
                    "canonical_node_id",
                    "source",
                    "content_hash",
                    "source_node_id",
                ),
                node_provenance_rows,
            )
            sync_edges(self._conn, edge_rows)
            sync_provenance(
                self._conn,
                "knowledge_base_edge_provenance",
                (
                    "canonical_edge_id",
                    "source",
                    "content_hash",
                    "source_edge_id",
                ),
                edge_provenance_rows,
            )

    def affected_ids(
        self,
        sources: Sequence[str],
    ) -> tuple[set[str], set[str]]:
        node_ids: set[str] = set()
        edge_ids: set[str] = set()
        for batch in _batches(sources):
            placeholders = ",".join("?" for _ in batch)
            node_ids.update(
                row["canonical_node_id"]
                for row in self._conn.execute(
                    "SELECT DISTINCT canonical_node_id "
                    "FROM knowledge_base_node_provenance "
                    f"WHERE source IN ({placeholders})",
                    batch,
                )
            )
            edge_ids.update(
                row["canonical_edge_id"]
                for row in self._conn.execute(
                    "SELECT DISTINCT canonical_edge_id "
                    "FROM knowledge_base_edge_provenance "
                    f"WHERE source IN ({placeholders})",
                    batch,
                )
            )
        for batch in _batches(sorted(node_ids)):
            placeholders = ",".join("?" for _ in batch)
            edge_ids.update(
                row["edge_id"]
                for row in self._conn.execute(
                    "SELECT edge_id FROM knowledge_base_edges "
                    f"WHERE source_node_id IN ({placeholders}) "
                    f"OR target_node_id IN ({placeholders})",
                    (*batch, *batch),
                )
            )
        return node_ids, edge_ids

    def apply_delta(self, delta: KnowledgeBaseDelta) -> None:
        node_ids = [node.id for node in delta.upserted_nodes]
        edge_ids = [edge.id for edge in delta.upserted_edges]
        with transaction(self._conn):
            self._delete_edges(delta.deleted_edge_ids)
            self._delete_nodes(delta.deleted_node_ids)
            self._conn.executemany(
                "INSERT INTO knowledge_base_nodes (node_id, payload) VALUES (?, ?) "
                "ON CONFLICT(node_id) DO UPDATE SET payload = excluded.payload "
                "WHERE payload <> excluded.payload",
                [
                    (
                        node.id,
                        node.model_copy(update={"provenance": []}).model_dump_json(),
                    )
                    for node in delta.upserted_nodes
                ],
            )
            self._conn.executemany(
                "DELETE FROM knowledge_base_node_aliases WHERE node_id = ?",
                [(node_id,) for node_id in node_ids],
            )
            self._conn.executemany(
                "INSERT INTO knowledge_base_node_aliases "
                "(node_id, alias, normalized_alias) VALUES (?, ?, ?)",
                [
                    (node.id, alias, normalized)
                    for node in delta.upserted_nodes
                    for alias in [node.label, *node.aliases]
                    if (normalized := normalized_entity_name(alias))
                ],
            )
            self._conn.executemany(
                "DELETE FROM knowledge_base_node_provenance "
                "WHERE canonical_node_id = ?",
                [(node_id,) for node_id in node_ids],
            )
            self._conn.executemany(
                "INSERT INTO knowledge_base_node_provenance "
                "(canonical_node_id, source, content_hash, source_node_id, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        node.id,
                        provenance.source,
                        provenance.content_hash,
                        provenance.node_id,
                        provenance.model_dump_json(),
                    )
                    for node in delta.upserted_nodes
                    for provenance in node.provenance
                ],
            )
            self._conn.executemany(
                "INSERT INTO knowledge_base_edges "
                "(edge_id, source_node_id, target_node_id, payload) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(edge_id) DO UPDATE SET "
                "source_node_id = excluded.source_node_id, "
                "target_node_id = excluded.target_node_id, "
                "payload = excluded.payload "
                "WHERE source_node_id <> excluded.source_node_id "
                "OR target_node_id <> excluded.target_node_id "
                "OR payload <> excluded.payload",
                [
                    (
                        edge.id,
                        edge.source,
                        edge.target,
                        edge.model_copy(
                            update={"provenance": []}
                        ).model_dump_json(),
                    )
                    for edge in delta.upserted_edges
                ],
            )
            self._conn.executemany(
                "DELETE FROM knowledge_base_edge_provenance "
                "WHERE canonical_edge_id = ?",
                [(edge_id,) for edge_id in edge_ids],
            )
            self._conn.executemany(
                "INSERT INTO knowledge_base_edge_provenance "
                "(canonical_edge_id, source, content_hash, source_edge_id, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        edge.id,
                        provenance.source,
                        provenance.content_hash,
                        provenance.edge_id,
                        provenance.model_dump_json(),
                    )
                    for edge in delta.upserted_edges
                    for provenance in edge.provenance
                ],
            )

    def _delete_nodes(self, node_ids: set[str]) -> None:
        rows = [(node_id,) for node_id in node_ids]
        self._conn.executemany(
            "DELETE FROM knowledge_base_node_aliases WHERE node_id = ?",
            rows,
        )
        self._conn.executemany(
            "DELETE FROM knowledge_base_node_provenance "
            "WHERE canonical_node_id = ?",
            rows,
        )
        self._conn.executemany(
            "DELETE FROM knowledge_base_nodes WHERE node_id = ?",
            rows,
        )

    def _delete_edges(self, edge_ids: set[str]) -> None:
        rows = [(edge_id,) for edge_id in edge_ids]
        self._conn.executemany(
            "DELETE FROM knowledge_base_edge_provenance "
            "WHERE canonical_edge_id = ?",
            rows,
        )
        self._conn.executemany(
            "DELETE FROM knowledge_base_edges WHERE edge_id = ?",
            rows,
        )

    def load(self) -> KnowledgeGraph:
        if not self.exists():
            raise FileNotFoundError("Knowledge base was not found.")
        return self._load_excluding_sources(())

    def load_excluding_sources(self, sources: Sequence[str]) -> KnowledgeGraph:
        if not self.exists():
            raise FileNotFoundError("Knowledge base was not found.")
        return self._load_excluding_sources(sources)

    def _load_excluding_sources(
        self,
        sources: Sequence[str],
    ) -> KnowledgeGraph:
        all_node_ids_with_provenance = {
            row["canonical_node_id"]
            for row in self._conn.execute(
                "SELECT DISTINCT canonical_node_id "
                "FROM knowledge_base_node_provenance"
            )
        }
        node_provenance = load_node_provenance(
            self._conn,
            excluded_sources=sources,
        )
        nodes = []
        for row in self._conn.execute(
            "SELECT node_id, payload FROM knowledge_base_nodes ORDER BY rowid"
        ):
            base = Node.model_validate_json(row["payload"])
            provenance = node_provenance.get(row["node_id"], [])
            if (
                row["node_id"] in all_node_ids_with_provenance
                and not provenance
            ):
                continue
            nodes.append(
                (
                    materialize_node(base.id, provenance)
                    if sources
                    else base.model_copy(update={"provenance": provenance})
                )
                if provenance
                else base
            )

        node_ids = {node.id for node in nodes}
        all_edge_ids_with_provenance = {
            row["canonical_edge_id"]
            for row in self._conn.execute(
                "SELECT DISTINCT canonical_edge_id "
                "FROM knowledge_base_edge_provenance"
            )
        }
        edge_provenance = load_edge_provenance(
            self._conn,
            excluded_sources=sources,
        )
        edges = []
        for row in self._conn.execute(
            "SELECT edge_id, payload FROM knowledge_base_edges ORDER BY rowid"
        ):
            base = Edge.model_validate_json(row["payload"])
            provenance = edge_provenance.get(row["edge_id"], [])
            if base.source not in node_ids or base.target not in node_ids:
                continue
            if (
                row["edge_id"] in all_edge_ids_with_provenance
                and not provenance
            ):
                continue
            edges.append(
                (
                    materialize_edge(
                        base.id,
                        base.source,
                        base.target,
                        provenance,
                    )
                    if sources
                    else base.model_copy(update={"provenance": provenance})
                )
                if provenance
                else base
            )
        return KnowledgeGraph(nodes=nodes, edges=edges)

    def exists(self) -> bool:
        row = self._conn.execute(
            "SELECT EXISTS("
            "SELECT 1 FROM knowledge_base_state WHERE singleton = 1 LIMIT 1)"
        ).fetchone()
        return bool(row[0])


def _batches(values: Sequence[str]) -> Iterable[tuple[str, ...]]:
    for start in range(0, len(values), _BATCH_SIZE):
        yield tuple(values[start : start + _BATCH_SIZE])
