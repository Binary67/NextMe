import logging

import pytest

from graphtool.ingestion.documents import load_documents
from graphtool.run_logging import LOGGER_NAME


def test_load_documents_returns_empty_for_missing_directory(tmp_path):
    documents = load_documents(
        tmp_path / "missing",
        source_root=tmp_path,
        pdf_llm=object(),
        pdf_cache_dir=tmp_path / "cache",
        presentation_cache_dir=tmp_path / "presentation-cache",
        audio_transcriber=object(),
        audio_corrector=object(),
        audio_cache_dir=tmp_path / "audio-cache",
        audio_transcription_terms=[],
    )

    assert documents == {}


def test_load_documents_reads_markdown_and_converts_pdf_and_pptx(
    caplog,
    monkeypatch,
    tmp_path,
):
    documents_dir = tmp_path / "documents"
    nested_dir = documents_dir / "guides"
    nested_dir.mkdir(parents=True)
    (documents_dir / "ignored.txt").write_text("ignored")
    (documents_dir / "a.MD").write_text("# A")
    pdf_path = nested_dir / "manual.PDF"
    pdf_path.write_bytes(b"pdf")
    pptx_path = nested_dir / "slides.PPTX"
    pptx_path.write_bytes(b"pptx")
    convert_calls = []
    presentation_calls = []

    def fake_convert(path, source, llm, cache_dir):
        convert_calls.append((path, source, llm, cache_dir))
        return "# Manual"

    monkeypatch.setattr(
        "graphtool.ingestion.documents.convert_pdf_to_markdown",
        fake_convert,
    )

    def fake_convert_pptx(path, source, llm, presentation_cache_dir, pdf_cache_dir):
        presentation_calls.append(
            (path, source, llm, presentation_cache_dir, pdf_cache_dir)
        )
        return "# Slides"

    monkeypatch.setattr(
        "graphtool.ingestion.documents.convert_pptx_to_markdown",
        fake_convert_pptx,
    )
    llm = object()
    cache_dir = tmp_path / "cache"
    presentation_cache_dir = tmp_path / "presentation-cache"
    logger = logging.getLogger(LOGGER_NAME)
    monkeypatch.setattr(logger, "propagate", True)
    caplog.set_level("INFO", logger=LOGGER_NAME)

    documents = load_documents(
        documents_dir,
        source_root=tmp_path,
        pdf_llm=llm,
        pdf_cache_dir=cache_dir,
        presentation_cache_dir=presentation_cache_dir,
        audio_transcriber=object(),
        audio_corrector=object(),
        audio_cache_dir=tmp_path / "audio-cache",
        audio_transcription_terms=[],
    )

    assert documents == {
        "documents/a.MD": "# A",
        "documents/guides/manual.PDF": "# Manual",
        "documents/guides/slides.PPTX": "# Slides",
    }
    assert convert_calls == [
        (pdf_path, "documents/guides/manual.PDF", llm, cache_dir)
    ]
    assert presentation_calls == [
        (
            pptx_path,
            "documents/guides/slides.PPTX",
            llm,
            presentation_cache_dir,
            cache_dir,
        )
    ]
    assert (
        "Found 3 supported documents: 1 Markdown, 1 PDF, 1 PowerPoint"
        in caplog.text
    )
    assert "[1/3] Reading Markdown: documents/a.MD" in caplog.text
    assert "[2/3] Processing PDF: documents/guides/manual.PDF" in caplog.text
    assert (
        "[3/3] Processing PowerPoint: documents/guides/slides.PPTX"
        in caplog.text
    )


def test_load_documents_does_not_return_partial_results_on_pdf_failure(
    monkeypatch,
    tmp_path,
):
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    (documents_dir / "guide.md").write_text("# Guide")
    (documents_dir / "manual.pdf").write_bytes(b"pdf")

    def fail_conversion(path, source, llm, cache_dir):
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(
        "graphtool.ingestion.documents.convert_pdf_to_markdown",
        fail_conversion,
    )

    with pytest.raises(RuntimeError, match="conversion failed"):
        load_documents(
            documents_dir,
            source_root=tmp_path,
            pdf_llm=object(),
            pdf_cache_dir=tmp_path / "cache",
            presentation_cache_dir=tmp_path / "presentation-cache",
            audio_transcriber=object(),
            audio_corrector=object(),
            audio_cache_dir=tmp_path / "audio-cache",
            audio_transcription_terms=[],
        )


def test_load_documents_converts_nested_audio(monkeypatch, tmp_path):
    documents_dir = tmp_path / "documents"
    recordings_dir = documents_dir / "recordings" / "interviews"
    recordings_dir.mkdir(parents=True)
    audio_path = recordings_dir / "customer.MP3"
    audio_path.write_bytes(b"audio")
    calls = []

    def fake_convert(path, source, transcriber, corrector, cache_dir, terms):
        calls.append((path, source, transcriber, corrector, cache_dir, terms))
        return "# Transcript: customer.MP3\n\nInterview text.\n"

    monkeypatch.setattr(
        "graphtool.ingestion.documents.convert_audio_to_markdown",
        fake_convert,
    )
    transcriber = object()
    corrector = object()
    cache_dir = tmp_path / "audio-cache"

    documents = load_documents(
        documents_dir,
        source_root=tmp_path,
        pdf_llm=object(),
        pdf_cache_dir=tmp_path / "pdf-cache",
        presentation_cache_dir=tmp_path / "presentation-cache",
        audio_transcriber=transcriber,
        audio_corrector=corrector,
        audio_cache_dir=cache_dir,
        audio_transcription_terms=["HIP-SA"],
    )

    assert documents == {
        "documents/recordings/interviews/customer.MP3": (
            "# Transcript: customer.MP3\n\nInterview text.\n"
        )
    }
    assert calls == [
        (
            audio_path,
            "documents/recordings/interviews/customer.MP3",
            transcriber,
            corrector,
            cache_dir,
            ["HIP-SA"],
        )
    ]
