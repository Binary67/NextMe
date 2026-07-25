import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field

from graphtool.chunking.types import Chunk
from graphtool.retrieval.bm25 import BM25Document, BM25Index
from graphtool.retrieval.types import ChunkHit

RECIPROCAL_RANK_CONSTANT = 60
MAX_RETURNED_HEADINGS = 20


class DocumentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    title: str
    headings: list[str] = Field(default_factory=list)
    chunk_count: int


class DocumentHit(DocumentRecord):
    score: float


class DocumentSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    documents: list[DocumentHit] = Field(default_factory=list)


@dataclass(frozen=True)
class PreparedDocumentRetriever:
    documents_by_source: dict[str, DocumentRecord]
    metadata_index: BM25Index

    def retrieve(
        self,
        query: str,
        *,
        passage_hits: Sequence[ChunkHit] = (),
        allowed_sources: set[str] | None = None,
        top_documents: int = 5,
    ) -> DocumentSearchResult:
        if top_documents < 1:
            raise ValueError("top_documents must be positive")

        sources = (
            set(self.documents_by_source)
            if allowed_sources is None
            else set(allowed_sources)
        )
        metadata_results = [
            (document, score)
            for document, score in self.metadata_index.rank(query)
            if score > 0 and document.id in sources
        ]
        metadata_ranks = {
            document.id: rank
            for rank, (document, _) in enumerate(metadata_results, start=1)
        }
        passage_ranks: dict[str, int] = {}
        for rank, hit in enumerate(passage_hits, start=1):
            source = hit.chunk.source
            if source in sources and source not in passage_ranks:
                passage_ranks[source] = rank

        query_key = _normalize(query)
        ranked = []
        for source in metadata_ranks.keys() | passage_ranks.keys():
            document = self.documents_by_source[source]
            explicit_match = _explicitly_names_document(query_key, document)
            score = 1.0 if explicit_match else 0.0
            if source in metadata_ranks:
                score += 1.0 / (
                    RECIPROCAL_RANK_CONSTANT + metadata_ranks[source]
                )
            if source in passage_ranks:
                score += 1.0 / (
                    RECIPROCAL_RANK_CONSTANT + passage_ranks[source]
                )
            ranked.append(
                DocumentHit(
                    **document.model_dump(exclude={"headings"}),
                    headings=document.headings[:MAX_RETURNED_HEADINGS],
                    score=score,
                )
            )

        ranked.sort(key=lambda item: (-item.score, item.source))
        return DocumentSearchResult(
            query=query,
            documents=ranked[:top_documents],
        )


def prepare_document_retriever(
    chunks: Sequence[Chunk],
) -> PreparedDocumentRetriever:
    chunks_by_source: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        chunks_by_source.setdefault(chunk.source, []).append(chunk)

    documents = [
        _document_record(source, source_chunks)
        for source, source_chunks in sorted(chunks_by_source.items())
    ]
    return PreparedDocumentRetriever(
        documents_by_source={
            document.source: document for document in documents
        },
        metadata_index=BM25Index(
            [
                BM25Document(
                    id=document.source,
                    text=_document_search_text(document),
                )
                for document in documents
            ]
        ),
    )


def _document_record(
    source: str,
    chunks: Sequence[Chunk],
) -> DocumentRecord:
    path = PurePosixPath(source)
    headings = []
    seen_headings = set()
    for chunk in chunks:
        heading = " > ".join(chunk.heading_path)
        normalized = _normalize(heading)
        if normalized and normalized not in seen_headings:
            headings.append(heading)
            seen_headings.add(normalized)
    return DocumentRecord(
        source=source,
        title=re.sub(r"[-_]+", " ", path.stem).strip(),
        headings=headings,
        chunk_count=len(chunks),
    )


def _document_search_text(document: DocumentRecord) -> str:
    path = PurePosixPath(document.source)
    return "\n".join(
        [
            document.source,
            path.name,
            path.stem,
            document.title,
            *document.headings,
        ]
    )


def _explicitly_names_document(
    query_key: str,
    document: DocumentRecord,
) -> bool:
    path = PurePosixPath(document.source)
    names = (
        _normalize(path.name),
        _normalize(document.title),
    )
    return any(name and name in query_key for name in names)


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))
