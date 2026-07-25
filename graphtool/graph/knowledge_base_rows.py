import sqlite3
from collections.abc import Sequence

from graphtool.graph.types import EdgeProvenance, NodeProvenance


def sync_aliases(
    conn: sqlite3.Connection,
    desired: dict[tuple[str, str], str],
) -> None:
    existing = {
        (row["node_id"], row["normalized_alias"]): row["alias"]
        for row in conn.execute(
            "SELECT node_id, normalized_alias, alias "
            "FROM knowledge_base_node_aliases"
        )
    }
    conn.executemany(
        "DELETE FROM knowledge_base_node_aliases "
        "WHERE node_id = ? AND normalized_alias = ?",
        list(set(existing) - set(desired)),
    )
    conn.executemany(
        "INSERT INTO knowledge_base_node_aliases "
        "(node_id, normalized_alias, alias) VALUES (?, ?, ?) "
        "ON CONFLICT(node_id, normalized_alias) DO UPDATE SET alias = excluded.alias "
        "WHERE alias <> excluded.alias",
        [
            (node_id, normalized, alias)
            for (node_id, normalized), alias in desired.items()
            if existing.get((node_id, normalized)) != alias
        ],
    )


def sync_edges(
    conn: sqlite3.Connection,
    desired: dict[str, tuple[str, str, str]],
) -> None:
    existing = {
        row["edge_id"]: (
            row["source_node_id"],
            row["target_node_id"],
            row["payload"],
        )
        for row in conn.execute(
            "SELECT edge_id, source_node_id, target_node_id, payload "
            "FROM knowledge_base_edges"
        )
    }
    conn.executemany(
        "DELETE FROM knowledge_base_edges WHERE edge_id = ?",
        [(edge_id,) for edge_id in set(existing) - set(desired)],
    )
    conn.executemany(
        "INSERT INTO knowledge_base_edges "
        "(edge_id, source_node_id, target_node_id, payload) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(edge_id) DO UPDATE SET "
        "source_node_id = excluded.source_node_id, "
        "target_node_id = excluded.target_node_id, "
        "payload = excluded.payload "
        "WHERE source_node_id <> excluded.source_node_id "
        "OR target_node_id <> excluded.target_node_id "
        "OR payload <> excluded.payload",
        [
            (edge_id, source, target, payload)
            for edge_id, (source, target, payload) in desired.items()
            if existing.get(edge_id) != (source, target, payload)
        ],
    )


def load_node_provenance(
    conn: sqlite3.Connection,
    node_id: str | None = None,
    *,
    excluded_sources: Sequence[str] = (),
) -> dict[str, list[NodeProvenance]]:
    query = (
        "SELECT canonical_node_id, payload "
        "FROM knowledge_base_node_provenance"
    )
    parameters: tuple[str, ...] = ()
    if node_id is not None:
        query += " WHERE canonical_node_id = ?"
        parameters = (node_id,)
    if excluded_sources:
        conjunction = " AND " if parameters else " WHERE "
        placeholders = ",".join("?" for _ in excluded_sources)
        query += f"{conjunction}source NOT IN ({placeholders})"
        parameters = (*parameters, *excluded_sources)
    query += " ORDER BY rowid"
    result: dict[str, list[NodeProvenance]] = {}
    for row in conn.execute(query, parameters):
        result.setdefault(row["canonical_node_id"], []).append(
            NodeProvenance.model_validate_json(row["payload"])
        )
    return result


def load_edge_provenance(
    conn: sqlite3.Connection,
    edge_id: str | None = None,
    *,
    excluded_sources: Sequence[str] = (),
) -> dict[str, list[EdgeProvenance]]:
    query = (
        "SELECT canonical_edge_id, payload "
        "FROM knowledge_base_edge_provenance"
    )
    parameters: tuple[str, ...] = ()
    if edge_id is not None:
        query += " WHERE canonical_edge_id = ?"
        parameters = (edge_id,)
    if excluded_sources:
        conjunction = " AND " if parameters else " WHERE "
        placeholders = ",".join("?" for _ in excluded_sources)
        query += f"{conjunction}source NOT IN ({placeholders})"
        parameters = (*parameters, *excluded_sources)
    query += " ORDER BY rowid"
    result: dict[str, list[EdgeProvenance]] = {}
    for row in conn.execute(query, parameters):
        result.setdefault(row["canonical_edge_id"], []).append(
            EdgeProvenance.model_validate_json(row["payload"])
        )
    return result
