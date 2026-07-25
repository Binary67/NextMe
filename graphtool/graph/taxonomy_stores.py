import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from graphtool.graph.taxonomy_types import (
    NodeTypeRegistry,
    TaxonomyPromotionRecord,
    TaxonomySuggestionRecord,
    default_node_type_registry,
)
from graphtool.storage import as_connection, transaction

_INSERT_SUGGESTION_SQL = (
    "INSERT INTO taxonomy_suggestions "
    "(suggested_type, normalized_suggested_type, node_id, "
    "node_label, current_type, source, chunk_id, created_at) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)


class JsonNodeTypeRegistryStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def save(self, registry: NodeTypeRegistry) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(registry.model_dump_json(indent=2))

    def load(self) -> NodeTypeRegistry:
        data = json.loads(self._path.read_text())
        return NodeTypeRegistry.model_validate(data)

    def load_or_default(self) -> NodeTypeRegistry:
        if not self.exists():
            return default_node_type_registry()
        return self.load()

    def exists(self) -> bool:
        return self._path.exists()


class SqliteTaxonomySuggestionStore:
    def __init__(self, conn_or_path: sqlite3.Connection | str | Path) -> None:
        self._conn = as_connection(conn_or_path)

    def append_many(self, records: Sequence[TaxonomySuggestionRecord]) -> None:
        if not records:
            return
        with transaction(self._conn):
            self._conn.executemany(
                _INSERT_SUGGESTION_SQL,
                [self._row_tuple(record) for record in records],
            )

    def save(self, records: Sequence[TaxonomySuggestionRecord]) -> None:
        with transaction(self._conn):
            self._conn.execute("DELETE FROM taxonomy_suggestions")
            self._conn.executemany(
                _INSERT_SUGGESTION_SQL,
                [self._row_tuple(record) for record in records],
            )

    def load(self) -> list[TaxonomySuggestionRecord]:
        rows = self._conn.execute(
            "SELECT suggested_type, normalized_suggested_type, node_id, "
            "node_label, current_type, source, chunk_id, created_at "
            "FROM taxonomy_suggestions ORDER BY rowid"
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def exists(self) -> bool:
        row = self._conn.execute(
            "SELECT EXISTS(SELECT 1 FROM taxonomy_suggestions LIMIT 1)"
        ).fetchone()
        return bool(row[0])

    def replace_source(
        self,
        source: str,
        records: Sequence[TaxonomySuggestionRecord],
    ) -> None:
        with transaction(self._conn):
            self._conn.execute(
                "DELETE FROM taxonomy_suggestions WHERE source = ?", (source,)
            )
            self._conn.executemany(
                _INSERT_SUGGESTION_SQL,
                [self._row_tuple(record) for record in records],
            )

    def delete_source(self, source: str) -> None:
        with transaction(self._conn):
            self._conn.execute(
                "DELETE FROM taxonomy_suggestions WHERE source = ?", (source,)
            )

    @staticmethod
    def _row_tuple(record: TaxonomySuggestionRecord) -> tuple:
        return (
            record.suggested_type,
            record.normalized_suggested_type,
            record.node_id,
            record.node_label,
            record.current_type,
            record.source,
            record.chunk_id,
            record.created_at.isoformat(),
        )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> TaxonomySuggestionRecord:
        return TaxonomySuggestionRecord(
            suggested_type=row["suggested_type"],
            normalized_suggested_type=row["normalized_suggested_type"],
            node_id=row["node_id"],
            node_label=row["node_label"],
            current_type=row["current_type"],
            source=row["source"],
            chunk_id=row["chunk_id"],
            created_at=row["created_at"],
        )


class JsonTaxonomyPromotionAuditStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def append_many(self, records: Sequence[TaxonomyPromotionRecord]) -> None:
        if not records:
            return
        existing = self.load()
        self.save([*existing, *records])

    def save(self, records: Sequence[TaxonomyPromotionRecord]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(
                [record.model_dump(mode="json") for record in records],
                indent=2,
                sort_keys=True,
            )
        )

    def load(self) -> list[TaxonomyPromotionRecord]:
        if not self.exists():
            return []
        data = json.loads(self._path.read_text())
        return [TaxonomyPromotionRecord.model_validate(item) for item in data]

    def exists(self) -> bool:
        return self._path.exists()
