import sqlite3
from collections.abc import Iterable


def sync_source_payloads(
    conn: sqlite3.Connection,
    table: str,
    id_column: str,
    source: str,
    rows: Iterable[tuple[str, str]],
) -> None:
    desired = dict(rows)
    existing = {
        row[id_column]: row["payload"]
        for row in conn.execute(
            f"SELECT {id_column}, payload FROM {table} WHERE source = ?",
            (source,),
        )
    }
    removed = set(existing) - set(desired)
    conn.executemany(
        f"DELETE FROM {table} WHERE source = ? AND {id_column} = ?",
        [(source, item_id) for item_id in removed],
    )
    conn.executemany(
        f"INSERT INTO {table} (source, {id_column}, payload) VALUES (?, ?, ?) "
        f"ON CONFLICT(source, {id_column}) DO UPDATE SET payload = excluded.payload "
        f"WHERE payload <> excluded.payload",
        [
            (source, item_id, payload)
            for item_id, payload in desired.items()
            if existing.get(item_id) != payload
        ],
    )


def sync_keyed_payloads(
    conn: sqlite3.Connection,
    table: str,
    id_column: str,
    desired: dict[str, str],
) -> None:
    existing = {
        row[id_column]: row["payload"]
        for row in conn.execute(f"SELECT {id_column}, payload FROM {table}")
    }
    conn.executemany(
        f"DELETE FROM {table} WHERE {id_column} = ?",
        [(item_id,) for item_id in set(existing) - set(desired)],
    )
    conn.executemany(
        f"INSERT INTO {table} ({id_column}, payload) VALUES (?, ?) "
        f"ON CONFLICT({id_column}) DO UPDATE SET payload = excluded.payload "
        f"WHERE payload <> excluded.payload",
        [
            (item_id, payload)
            for item_id, payload in desired.items()
            if existing.get(item_id) != payload
        ],
    )


def sync_provenance(
    conn: sqlite3.Connection,
    table: str,
    columns: tuple[str, str, str, str],
    desired: dict[tuple[str, str, str, str], str],
) -> None:
    column_list = ", ".join(columns)
    existing = {
        tuple(row[column] for column in columns): row["payload"]
        for row in conn.execute(f"SELECT {column_list}, payload FROM {table}")
    }
    where = " AND ".join(f"{column} = ?" for column in columns)
    conn.executemany(
        f"DELETE FROM {table} WHERE {where}",
        list(set(existing) - set(desired)),
    )
    conflict = ", ".join(columns)
    placeholders = ", ".join("?" for _ in range(5))
    conn.executemany(
        f"INSERT INTO {table} ({column_list}, payload) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT({conflict}) DO UPDATE SET payload = excluded.payload "
        f"WHERE payload <> excluded.payload",
        [
            (*key, payload)
            for key, payload in desired.items()
            if existing.get(key) != payload
        ],
    )
