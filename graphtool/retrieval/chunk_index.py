from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from graphtool.chunking.types import Chunk
from graphtool.graph.types import Edge, Node
from graphtool.llm.base import EmbeddingClient
from graphtool.retrieval.context import (
    ChunkGraphIndex,
    node_text,
    properties_text,
    relationship_text,
)
from graphtool.retrieval.embedding_store import (
    ChunkEmbeddingRecord,
    ChunkEmbeddingStore,
    chunk_embedding_input_hash,
)


@dataclass(frozen=True)
class ChunkSearchFields:
    primary_labels: str
    aliases: str
    content: str
    metadata: str


def search_fields_by_chunk(
    chunks_by_id: dict[str, Chunk],
    index: ChunkGraphIndex,
) -> dict[str, ChunkSearchFields]:
    fields_by_chunk = {}
    for chunk in chunks_by_id.values():
        source_path = PurePosixPath(chunk.source)
        nodes = index.nodes_by_chunk.get(chunk.id, [])
        edges = index.edges_by_chunk.get(chunk.id, [])
        fields_by_chunk[chunk.id] = ChunkSearchFields(
            primary_labels="\n".join(
                _unique_search_text(node.label for node in nodes)
            ),
            aliases="\n".join(
                _unique_search_text(
                    alias for node in nodes for alias in node.aliases
                )
            ),
            content=chunk.text,
            metadata="\n".join(
                _unique_search_text(
                    [
                        chunk.source,
                        source_path.name,
                        source_path.stem,
                        *chunk.heading_path,
                        *(node.type for node in nodes),
                        *(
                            node.suggested_type
                            for node in nodes
                            if node.suggested_type is not None
                        ),
                        *(
                            properties_text(node.properties)
                            for node in nodes
                            if node.properties
                        ),
                        *(_relationship_metadata_text(edge) for edge in edges),
                    ]
                )
            ),
        )
    return fields_by_chunk


def searchable_text_by_chunk(
    chunks_by_id: dict[str, Chunk],
    index: ChunkGraphIndex,
    nodes_by_id: dict[str, Node],
) -> dict[str, str]:
    return {
        chunk.id: _chunk_text(
            chunk,
            index.nodes_by_chunk.get(chunk.id, []),
            index.edges_by_chunk.get(chunk.id, []),
            nodes_by_id,
        )
        for chunk in chunks_by_id.values()
    }


def prepare_chunk_vectors(
    searchable_text: dict[str, str],
    embedding_client: EmbeddingClient | None,
    chunk_embedding_store: ChunkEmbeddingStore | None,
) -> dict[str, list[float]]:
    if embedding_client is None or not searchable_text:
        return {}

    records = (
        chunk_embedding_store.load()
        if chunk_embedding_store is not None
        else {}
    )
    embedding_model = embedding_client.embedding_model
    records_to_save = dict(records)
    chunk_records: dict[str, ChunkEmbeddingRecord] = {}
    missing: list[tuple[str, str, str]] = []

    for chunk_id, text in searchable_text.items():
        text_hash = chunk_embedding_input_hash(text)
        record = records.get(chunk_id)
        if (
            record is not None
            and record.embedding_model == embedding_model
            and record.embedding_input_hash == text_hash
        ):
            chunk_records[chunk_id] = record
        else:
            missing.append((chunk_id, text_hash, text))

    if missing:
        vectors = embedding_client.embed_texts(
            [text for _, _, text in missing]
        )
        for (chunk_id, text_hash, _), vector in zip(
            missing,
            vectors,
            strict=True,
        ):
            record = ChunkEmbeddingRecord(
                chunk_id=chunk_id,
                embedding_model=embedding_model,
                embedding_input_hash=text_hash,
                vector=vector,
            )
            records_to_save[chunk_id] = record
            chunk_records[chunk_id] = record
        if chunk_embedding_store is not None:
            chunk_embedding_store.upsert(
                {
                    chunk_id: records_to_save[chunk_id]
                    for chunk_id, _, _ in missing
                }
            )

    return {
        chunk_id: record.vector
        for chunk_id, record in chunk_records.items()
    }


def _chunk_text(
    chunk: Chunk,
    nodes: Sequence[Node],
    edges: Sequence[Edge],
    nodes_by_id: dict[str, Node],
) -> str:
    lines = ["Source:", chunk.source]
    if chunk.heading_path:
        lines.extend(["Heading:", " > ".join(chunk.heading_path)])
    lines.extend(["Content:", chunk.text])
    if nodes:
        lines.append("Entities:")
        lines.extend(node_text(node) for node in nodes)
    if edges:
        lines.append("Relationships:")
        lines.extend(
            relationship_text(
                edge,
                nodes_by_id[edge.source],
                nodes_by_id[edge.target],
            )
            for edge in edges
        )
    return "\n".join(lines)


def _relationship_metadata_text(edge: Edge) -> str:
    if not edge.properties:
        return edge.label
    return f"{edge.label} | properties: {properties_text(edge.properties)}"


def _unique_search_text(values: Iterable[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        normalized = " ".join(value.casefold().split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(value)
    return unique
