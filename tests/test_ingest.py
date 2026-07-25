from types import SimpleNamespace
from unittest.mock import Mock

import ingest as ingest_module


class FakeRuntime:
    def __init__(self, paths) -> None:
        self.paths = paths
        self.corpus_stores = object()
        self.graph_store = object()
        self.chunk_store = object()
        self.fast_llm = object()
        self.audio_transcriber = object()
        self.knowledge_base_store = object()
        self.graph_embedding_store = object()
        self.knowledge_base_embedding_store = object()
        self.chunk_embedding_store = object()
        self.chunk_extraction_store = object()
        self.taxonomy_suggestion_store = object()


def test_main_synchronizes_documents_and_exports_visualizations(
    monkeypatch,
    capsys,
    tmp_path,
):
    paths = SimpleNamespace(
        root=tmp_path,
        documents_dir=tmp_path / "documents",
        pdf_conversions_dir=tmp_path / "pdf-conversions",
        presentation_conversions_dir=tmp_path / "presentation-conversions",
        audio_transcriptions_dir=tmp_path / "audio-transcriptions",
        audio_transcription_glossary_path=(
            tmp_path / "config" / "transcription_glossary.json"
        ),
        dropped_edges_path=tmp_path / "dropped_edges.jsonl",
        logs_dir=tmp_path / "logs",
        visualizations_dir=tmp_path / "visualizations",
    )
    config = SimpleNamespace(entity_resolution_min_candidate_similarity=0.8)
    runtime = FakeRuntime(paths)
    paths.audio_transcription_glossary_path.parent.mkdir()
    paths.audio_transcription_glossary_path.write_text('{"terms": []}')
    logger = Mock()
    visualization_path = paths.visualizations_dir / "knowledge-base.html"

    monkeypatch.setattr(ingest_module, "default_paths", Mock(return_value=paths))
    monkeypatch.setattr(
        ingest_module,
        "configure_run_logger",
        Mock(return_value=logger),
    )
    monkeypatch.setattr(
        ingest_module,
        "load_azure_openai_config",
        Mock(return_value=config),
    )
    monkeypatch.setattr(
        ingest_module,
        "create_runtime",
        Mock(return_value=runtime),
    )
    monkeypatch.setattr(
        ingest_module,
        "load_documents",
        Mock(return_value={"docs/guide.md": "# Guide\nText."}),
    )
    monkeypatch.setattr(
        ingest_module,
        "synchronize_documents",
        Mock(
            return_value=SimpleNamespace(
                added_sources=["docs/guide.md"],
                changed_sources=[],
                deleted_sources=[],
                unchanged_sources=[],
            )
        ),
    )
    monkeypatch.setattr(
        ingest_module,
        "export_knowledge_base_visualizations",
        Mock(return_value=[visualization_path]),
    )

    ingest_module.main()

    output = capsys.readouterr().out
    ingest_module.load_documents.assert_called_once_with(
        paths.documents_dir,
        source_root=paths.root,
        pdf_llm=runtime.fast_llm,
        pdf_cache_dir=paths.pdf_conversions_dir,
        presentation_cache_dir=paths.presentation_conversions_dir,
        audio_transcriber=runtime.audio_transcriber,
        audio_corrector=runtime.fast_llm,
        audio_cache_dir=paths.audio_transcriptions_dir,
        audio_transcription_terms=[],
    )
    ingest_module.synchronize_documents.assert_called_once_with(
        {"docs/guide.md": "# Guide\nText."},
        runtime.corpus_stores,
        runtime.fast_llm,
        chunk_extraction_store=runtime.chunk_extraction_store,
        dropped_edges_path=paths.dropped_edges_path,
        min_candidate_similarity=0.8,
    )
    ingest_module.export_knowledge_base_visualizations.assert_called_once_with(
        runtime.graph_store,
        paths.visualizations_dir,
        knowledge_base_store=runtime.knowledge_base_store,
    )
    assert f"- {visualization_path}" in output
