from graphtool.chunking.types import Chunk
from graphtool.retrieval.documents import prepare_document_retriever
from graphtool.retrieval.types import ChunkHit


def _chunk(
    source: str,
    chunk_id: str,
    text: str,
    heading: str,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        source=source,
        index=0,
        text=text,
        heading_path=[heading],
    )


def _hit(chunk: Chunk) -> ChunkHit:
    return ChunkHit(
        chunk=chunk,
        score=1.0,
        relevance=1.0,
        linked_nodes=[],
        linked_relationships=[],
    )


def test_document_search_matches_filename_without_filename_in_content():
    phoenix = _chunk(
        "documents/work/phoenix-architecture.pdf",
        "phoenix-0000",
        "PostgreSQL stores application records.",
        "Storage",
    )
    atlas = _chunk(
        "documents/work/atlas-architecture.pdf",
        "atlas-0000",
        "DynamoDB stores application records.",
        "Storage",
    )
    retriever = prepare_document_retriever([phoenix, atlas])

    result = retriever.retrieve("Summarize Phoenix Architecture")

    assert result.documents[0].source == phoenix.source
    assert result.documents[0].title == "phoenix architecture"
    assert result.documents[0].headings == ["Storage"]


def test_document_search_uses_passage_hits_for_topic_discovery():
    migration = _chunk(
        "documents/database-runbook.md",
        "migration-0000",
        "Rotate credentials after migrating the database.",
        "Operations",
    )
    unrelated = _chunk(
        "documents/employee-handbook.md",
        "handbook-0000",
        "Employees receive annual leave.",
        "Benefits",
    )
    retriever = prepare_document_retriever([migration, unrelated])

    result = retriever.retrieve(
        "credential rotation",
        passage_hits=[_hit(migration)],
    )

    assert [document.source for document in result.documents] == [
        migration.source
    ]


def test_document_search_respects_allowed_sources():
    work = _chunk(
        "documents/work/project-plan.md",
        "work-0000",
        "The project launches in June.",
        "Schedule",
    )
    personal = _chunk(
        "documents/personal/project-plan.md",
        "personal-0000",
        "The personal project launches in July.",
        "Schedule",
    )
    retriever = prepare_document_retriever([work, personal])

    result = retriever.retrieve(
        "project plan",
        passage_hits=[_hit(personal), _hit(work)],
        allowed_sources={work.source},
    )

    assert [document.source for document in result.documents] == [work.source]
