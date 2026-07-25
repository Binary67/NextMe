import hashlib
import json
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial

from pydantic import ValidationError

from graphtool.chunking.types import Chunk
from graphtool.graph.extraction_store import (
    ChunkExtractionStore,
    ExtractedKnowledgeGraph,
)
from graphtool.graph.taxonomy_types import (
    UNCLASSIFIED_NODE_TYPE,
    canonical_node_type_text,
)
from graphtool.llm.base import LLMClient
from graphtool.llm.types import LLMMessage
from graphtool.run_logging import LOGGER_NAME

RUN_LOGGER = logging.getLogger(LOGGER_NAME)

SYSTEM_PROMPT = (
    "You extract compact knowledge graphs from markdown content. Identify only "
    "important domain entities as nodes and meaningful relationships as edges. "
    "Do not create nodes for prompt metadata, chunks, source file paths, "
    "markdown headings, tables, rows, columns, URLs, or generic document "
    "structure unless the content is explicitly about those concepts. Table "
    "contents can contain useful facts; extract those facts, not the table "
    "mechanics. Prefer a small graph of the most salient entities. Assign every "
    "node a unique temporary ref within this response. Every edge must use "
    "source_ref and target_ref to reference existing node refs. Node type must "
    "be one of: "
    f"{canonical_node_type_text()}. If none of those types fit, use "
    f"{UNCLASSIFIED_NODE_TYPE} and provide suggested_type with the missing "
    "taxonomy type. Return only the structured nodes and edges."
)

USER_PROMPT_TEMPLATE = (
    "Extract a compact knowledge graph from the markdown content below.\n\n"
    "Context only, do not extract this as graph content:\n"
    "Heading path: {heading_path}\n\n"
    "Markdown content:\n"
    "{markdown}"
)


@dataclass(frozen=True)
class ChunkExtractions:
    graphs: list[ExtractedKnowledgeGraph]
    records: dict[str, ExtractedKnowledgeGraph]
    cached_chunks: int
    generated_chunks: int
    extraction_requests: int


def extract_chunks(
    chunks: Sequence[Chunk],
    source: str,
    llm: LLMClient,
    extraction_store: ChunkExtractionStore | None,
    *,
    max_workers: int,
) -> ChunkExtractions:
    messages_by_chunk = [chunk_messages(chunk) for chunk in chunks]
    extract = partial(_extract_chunk, llm=llm)

    if extraction_store is None:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            graphs = list(executor.map(extract, chunks, messages_by_chunk))
        return ChunkExtractions(
            graphs=graphs,
            records={},
            cached_chunks=0,
            generated_chunks=len(chunks),
            extraction_requests=len(chunks),
        )

    cached_records = extraction_store.load(source)
    cache_keys = [
        extraction_cache_key(messages, llm.text_model)
        for messages in messages_by_chunk
    ]
    records = {
        cache_key: cached_records[cache_key]
        for cache_key in dict.fromkeys(cache_keys)
        if cache_key in cached_records
    }
    missing = {}
    for cache_key, chunk, messages in zip(
        cache_keys,
        chunks,
        messages_by_chunk,
        strict=True,
    ):
        if cache_key not in records and cache_key not in missing:
            missing[cache_key] = (chunk, messages)

    missing_items = list(missing.items())
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        generated = list(
            executor.map(
                extract,
                (item[1][0] for item in missing_items),
                (item[1][1] for item in missing_items),
            )
        )
    records.update(
        (item[0], graph)
        for item, graph in zip(missing_items, generated, strict=True)
    )
    cached_chunks = sum(cache_key in cached_records for cache_key in cache_keys)
    return ChunkExtractions(
        graphs=[records[cache_key] for cache_key in cache_keys],
        records={
            cache_key: records[cache_key]
            for cache_key in dict.fromkeys(cache_keys)
        },
        cached_chunks=cached_chunks,
        generated_chunks=len(chunks) - cached_chunks,
        extraction_requests=len(missing_items),
    )


def chunk_messages(chunk: Chunk) -> list[LLMMessage]:
    return [
        LLMMessage(role="system", content=SYSTEM_PROMPT),
        LLMMessage(
            role="user",
            content=USER_PROMPT_TEMPLATE.format(
                heading_path=_heading_path_text(chunk),
                markdown=chunk.text,
            ),
        ),
    ]


def extraction_cache_key(
    messages: Sequence[LLMMessage],
    text_model: str,
) -> str:
    payload = {
        "messages": [
            {"role": message.role, "content": message.content}
            for message in messages
        ],
        "response_schema": ExtractedKnowledgeGraph.model_json_schema(),
        "text_model": text_model,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _extract_chunk(
    chunk: Chunk,
    messages: Sequence[LLMMessage],
    *,
    llm: LLMClient,
) -> ExtractedKnowledgeGraph:
    try:
        graph = llm.generate_structured(messages, ExtractedKnowledgeGraph)
    except ValidationError:
        RUN_LOGGER.warning(
            "Retrying chunk graph generation after invalid structured response: %s",
            chunk.id,
        )
        graph = llm.generate_structured(messages, ExtractedKnowledgeGraph)
    return graph


def _heading_path_text(chunk: Chunk) -> str:
    if not chunk.heading_path:
        return "(none)"
    return " > ".join(chunk.heading_path)
