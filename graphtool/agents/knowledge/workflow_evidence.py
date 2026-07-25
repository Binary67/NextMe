from graphtool.agents.knowledge.state import (
    AgentChunkReference,
    AgentState,
    DocumentEvidenceRecord,
    EvidenceRecord,
    GraphPathEvidenceRecord,
)
from graphtool.retrieval import SourceReference


def chunk_key(source: str, chunk_id: str) -> str:
    return f"{source} :: {chunk_id}"


def chunk_source_reference(chunk: AgentChunkReference) -> SourceReference:
    return SourceReference(
        source=chunk.source,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
    )


def neighborhood_evidence_text(chunk) -> str:
    heading = " > ".join(chunk.heading_path)
    metadata = f"{chunk.chunk_id} | {chunk.source}"
    if chunk.page_start is not None:
        metadata = f"{metadata} | pages {chunk.page_start}-{chunk.page_end}"
    if heading:
        metadata = f"{metadata} | {heading}"
    return f"[{metadata}]\n{chunk.text}"


def has_current_evidence(state: AgentState) -> bool:
    subquestion_index = state["subquestion_index"]
    return any(
        subquestion_index in record.subquestion_indexes
        for record in state["evidence"]
    ) or any(
        subquestion_index in record.subquestion_indexes
        for record in state["document_evidence"]
    )


def merge_allowed_chunks(
    existing: list[AgentChunkReference],
    incoming: list[AgentChunkReference],
) -> list[AgentChunkReference]:
    merged = list(existing)
    keys = {chunk_key(item.source, item.chunk_id) for item in existing}
    for item in incoming:
        key = chunk_key(item.source, item.chunk_id)
        if key not in keys:
            merged.append(item)
            keys.add(key)
    return merged


def merge_evidence_record(
    existing: list[EvidenceRecord],
    incoming: EvidenceRecord,
    subquestion_index: int,
) -> tuple[list[EvidenceRecord], bool]:
    merged = list(existing)
    key = chunk_key(incoming.source, incoming.chunk_id)
    for index, record in enumerate(merged):
        if chunk_key(record.source, record.chunk_id) != key:
            continue
        if subquestion_index not in record.subquestion_indexes:
            merged[index] = record.model_copy(
                update={
                    "subquestion_indexes": [
                        *record.subquestion_indexes,
                        subquestion_index,
                    ]
                }
            )
        return merged, False
    merged.append(incoming)
    return merged, True


def merge_document_evidence_record(
    existing: list[DocumentEvidenceRecord],
    incoming: DocumentEvidenceRecord,
    subquestion_index: int,
) -> tuple[list[DocumentEvidenceRecord], bool]:
    merged = list(existing)
    for index, record in enumerate(merged):
        if (
            record.source != incoming.source
            or record.query.casefold() != incoming.query.casefold()
        ):
            continue
        if subquestion_index not in record.subquestion_indexes:
            merged[index] = record.model_copy(
                update={
                    "subquestion_indexes": [
                        *record.subquestion_indexes,
                        subquestion_index,
                    ]
                }
            )
        return merged, False
    merged.append(incoming)
    return merged, True


def merge_graph_path_evidence_record(
    existing: list[GraphPathEvidenceRecord],
    incoming: GraphPathEvidenceRecord,
    subquestion_index: int,
) -> tuple[list[GraphPathEvidenceRecord], bool]:
    merged = list(existing)
    key = (tuple(incoming.node_ids), tuple(incoming.edge_ids))
    for index, record in enumerate(merged):
        if (tuple(record.node_ids), tuple(record.edge_ids)) != key:
            continue
        if subquestion_index not in record.subquestion_indexes:
            merged[index] = record.model_copy(
                update={
                    "subquestion_indexes": [
                        *record.subquestion_indexes,
                        subquestion_index,
                    ]
                }
            )
        return merged, False
    merged.append(incoming)
    return merged, True
