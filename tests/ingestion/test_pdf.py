import json
import logging

import pytest
from pypdf import PdfWriter

from graphtool.ingestion import (
    pdf,
    pdf_cache,
    pdf_conversion,
    pdf_pages,
    pdf_prompts,
)
from graphtool.ingestion.pdf import convert_pdf_to_markdown
from graphtool.ingestion.pdf_conversion import (
    ConvertedPdfPage,
    PdfBatchConversion,
    PdfPageConversion,
)
from graphtool.llm.types import LLMImageContent, LLMTextContent
from graphtool.run_logging import LOGGER_NAME
from graphtool.source import source_key


class FakeLLM:
    def __init__(self, responses, *, text_model="fast-deployment"):
        self.text_model = text_model
        self.responses = list(responses)
        self.calls = []

    def generate_structured(self, messages, response_model):
        self.calls.append((list(messages), response_model))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _conversion(*pages, ending_heading_path=None):
    return PdfBatchConversion(
        pages=[
            ConvertedPdfPage(page_number=number, markdown=markdown)
            for number, markdown in pages
        ],
        ending_heading_path=ending_heading_path or [],
    )


def _prepare(monkeypatch, page_texts):
    monkeypatch.setattr(pdf, "extract_page_texts", lambda path, source: page_texts)
    monkeypatch.setattr(pdf.shutil, "which", lambda name: "/usr/bin/pdftoppm")
    render_calls = []

    def fake_render(path, page_numbers, pdftoppm, source):
        render_calls.append(list(page_numbers))
        return [f"page-{number}".encode() for number in page_numbers]

    monkeypatch.setattr(pdf, "render_pages", fake_render)
    return render_calls


def test_convert_pdf_batches_pages_and_assembles_canonical_markdown(
    monkeypatch,
    tmp_path,
):
    original_assemble_markdown = pdf_conversion.assemble_markdown
    assembled_page_counts = []

    def track_assembled_pages(pages):
        assembled_page_counts.append(len(pages))
        return original_assemble_markdown(pages)

    monkeypatch.setattr(pdf_conversion, "assemble_markdown", track_assembled_pages)
    monkeypatch.setattr(pdf, "assemble_markdown", track_assembled_pages)
    render_calls = _prepare(monkeypatch, ["First text", "Second text", "Third text"])
    path = tmp_path / "manual.pdf"
    path.write_bytes(b"pdf")
    llm = FakeLLM(
        [
            _conversion(
                (1, "# Manual\r\n\r\nFirst."),
                (2, "## Setup\nSecond."),
                ending_heading_path=["Manual", "Setup"],
            ),
            _conversion((3, "Third.")),
        ]
    )

    markdown = convert_pdf_to_markdown(
        path,
        "documents/manual.pdf",
        llm,
        tmp_path / "cache",
    )

    assert render_calls == [[1, 2], [3]]
    assert assembled_page_counts == [2, 1, 3]
    assert markdown == (
        "<!-- graphtool:page=1 -->\n\n# Manual\n\nFirst.\n\n"
        "<!-- graphtool:page=2 -->\n\n## Setup\nSecond.\n\n"
        "<!-- graphtool:page=3 -->\n\nThird.\n"
    )
    assert [call[1] for call in llm.calls] == [PdfBatchConversion] * 2

    first_parts = llm.calls[0][0][1].content
    assert isinstance(first_parts[0], LLMTextContent)
    assert isinstance(first_parts[2], LLMImageContent)
    assert first_parts[2].detail == "high"
    second_context = llm.calls[1][0][1].content[0].text
    assert "['Manual', 'Setup']" in second_context
    first_batch_markdown = (
        "<!-- graphtool:page=1 -->\n\n# Manual\n\nFirst.\n\n"
        "<!-- graphtool:page=2 -->\n\n## Setup\nSecond.\n"
    )
    assert second_context.endswith(
        first_batch_markdown[-pdf_conversion.PDF_CONTEXT_TAIL_CHARS :]
    )


def test_convert_pdf_keeps_exact_bounded_context_tail(monkeypatch, tmp_path):
    monkeypatch.setattr(pdf_pages, "PDF_BATCH_MAX_PAGES", 1)
    monkeypatch.setattr(pdf_conversion, "PDF_CONTEXT_TAIL_CHARS", 40)
    _prepare(monkeypatch, ["One", "Two", "Three"])
    path = tmp_path / "manual.pdf"
    path.write_bytes(b"pdf")
    pages = [
        ConvertedPdfPage(page_number=1, markdown="A" * 50),
        ConvertedPdfPage(page_number=2, markdown="B" * 50),
        ConvertedPdfPage(page_number=3, markdown="C" * 50),
    ]
    llm = FakeLLM(
        [
            PdfBatchConversion(pages=[page])
            for page in pages
        ]
    )

    markdown = convert_pdf_to_markdown(
        path,
        "documents/manual.pdf",
        llm,
        tmp_path / "cache",
    )

    context_prefix = "context only; do not repeat it:\n\n"
    second_context = llm.calls[1][0][1].content[0].text.split(
        context_prefix,
        maxsplit=1,
    )[1]
    third_context = llm.calls[2][0][1].content[0].text.split(
        context_prefix,
        maxsplit=1,
    )[1]
    assert second_context == pdf.assemble_markdown(pages[:1])[-40:]
    assert third_context == pdf.assemble_markdown(pages[:2])[-40:]
    assert markdown == pdf.assemble_markdown(pages)


def test_convert_pdf_uses_complete_cache_without_rendering_or_llm(
    caplog,
    monkeypatch,
    tmp_path,
):
    render_calls = _prepare(monkeypatch, ["Text"])
    path = tmp_path / "manual.pdf"
    path.write_bytes(b"pdf")
    first_llm = FakeLLM([_conversion((1, "# Cached"))])
    cache_dir = tmp_path / "cache"
    logger = logging.getLogger(LOGGER_NAME)
    monkeypatch.setattr(logger, "propagate", True)
    caplog.set_level("INFO", logger=LOGGER_NAME)

    expected = convert_pdf_to_markdown(
        path,
        "documents/manual.pdf",
        first_llm,
        cache_dir,
    )
    monkeypatch.setattr(
        pdf,
        "extract_page_texts",
        lambda path, source: pytest.fail("completed cache parsed the PDF"),
    )
    monkeypatch.setattr(
        pdf.shutil,
        "which",
        lambda name: pytest.fail("completed cache looked up Poppler"),
    )
    second_llm = FakeLLM([])
    actual = convert_pdf_to_markdown(
        path,
        "documents/manual.pdf",
        second_llm,
        cache_dir,
    )

    assert actual == expected
    assert render_calls == [[1]]
    assert second_llm.calls == []
    assert "Converting content for documents/manual.pdf (1 page)" in caplog.text
    assert "Processed documents/manual.pdf: page 1 of 1" in caplog.text
    assert "Finished content conversion for documents/manual.pdf" in caplog.text
    assert (
        "Using cached content conversion for documents/manual.pdf (1 page)"
        in caplog.text
    )


def test_convert_pdf_resumes_validated_batches_after_failure(monkeypatch, tmp_path):
    render_calls = _prepare(monkeypatch, ["One", "Two", "Three"])
    path = tmp_path / "manual.pdf"
    path.write_bytes(b"pdf")
    cache_dir = tmp_path / "cache"
    failing_llm = FakeLLM(
        [
            _conversion((1, "One."), (2, "Two.")),
            RuntimeError("request failed"),
        ]
    )

    with pytest.raises(RuntimeError, match="request failed"):
        convert_pdf_to_markdown(
            path,
            "documents/manual.pdf",
            failing_llm,
            cache_dir,
        )

    resumed_llm = FakeLLM([_conversion((3, "Three."))])
    markdown = convert_pdf_to_markdown(
        path,
        "documents/manual.pdf",
        resumed_llm,
        cache_dir,
    )

    assert render_calls == [[1, 2], [3], [3]]
    assert len(resumed_llm.calls) == 1
    resumed_context = resumed_llm.calls[0][0][1].content[0].text
    cached_markdown = (
        "<!-- graphtool:page=1 -->\n\nOne.\n\n"
        "<!-- graphtool:page=2 -->\n\nTwo.\n"
    )
    assert resumed_context.endswith(
        cached_markdown[-pdf_conversion.PDF_CONTEXT_TAIL_CHARS :]
    )
    assert "Three." in markdown


def test_convert_pdf_invalidates_cache_when_source_changes(monkeypatch, tmp_path):
    render_calls = _prepare(monkeypatch, ["Text"])
    path = tmp_path / "manual.pdf"
    cache_dir = tmp_path / "cache"
    path.write_bytes(b"first")
    convert_pdf_to_markdown(
        path,
        "documents/manual.pdf",
        FakeLLM([_conversion((1, "First."))]),
        cache_dir,
    )
    path.write_bytes(b"second")

    markdown = convert_pdf_to_markdown(
        path,
        "documents/manual.pdf",
        FakeLLM([_conversion((1, "Second."))]),
        cache_dir,
    )

    assert render_calls == [[1], [1]]
    assert "Second." in markdown


def test_convert_pdf_invalidates_cache_when_fast_model_changes(
    monkeypatch,
    tmp_path,
):
    render_calls = _prepare(monkeypatch, ["Text"])
    path = tmp_path / "manual.pdf"
    path.write_bytes(b"pdf")
    cache_dir = tmp_path / "cache"
    convert_pdf_to_markdown(
        path,
        "documents/manual.pdf",
        FakeLLM([_conversion((1, "First."))], text_model="fast-a"),
        cache_dir,
    )

    markdown = convert_pdf_to_markdown(
        path,
        "documents/manual.pdf",
        FakeLLM([_conversion((1, "Second."))], text_model="fast-b"),
        cache_dir,
    )

    assert render_calls == [[1], [1]]
    assert "Second." in markdown


def test_convert_pdf_retries_invalid_batch_with_correction(
    monkeypatch,
    caplog,
    tmp_path,
):
    _prepare(monkeypatch, ["One", "Two"])
    path = tmp_path / "manual.pdf"
    path.write_bytes(b"pdf")
    llm = FakeLLM(
        [
            _conversion((2, "Two.")),
            _conversion((1, "One."), (2, "Two.")),
        ]
    )

    with caplog.at_level("INFO", logger=LOGGER_NAME):
        markdown = convert_pdf_to_markdown(
            path,
            "documents/manual.pdf",
            llm,
            tmp_path / "cache",
        )

    assert "One." in markdown
    assert "Two." in markdown
    assert [call[1] for call in llm.calls] == [
        PdfBatchConversion,
        PdfBatchConversion,
    ]
    correction = llm.calls[1][0][1].content[0].text
    assert "expected pages [1, 2], received [2]" in correction
    assert "exact order: [1, 2]" in correction
    assert "Retrying PDF batch conversion" in caplog.text
    assert "Recovered PDF batch conversion on retry" in caplog.text


def test_convert_pdf_falls_back_to_caller_numbered_individual_pages(
    monkeypatch,
    caplog,
    tmp_path,
):
    _prepare(monkeypatch, ["One", "Two"])
    path = tmp_path / "manual.pdf"
    path.write_bytes(b"pdf")
    llm = FakeLLM(
        [
            _conversion((2, "Two.")),
            _conversion((1, "One."), (1, "Duplicate.")),
            PdfPageConversion(
                markdown="# One",
                ending_heading_path=["One"],
            ),
            PdfPageConversion(
                markdown="Two.",
                ending_heading_path=["One"],
            ),
        ]
    )

    with caplog.at_level("INFO", logger=LOGGER_NAME):
        markdown = convert_pdf_to_markdown(
            path,
            "documents/manual.pdf",
            llm,
            tmp_path / "cache",
        )

    assert markdown == (
        "<!-- graphtool:page=1 -->\n\n# One\n\n"
        "<!-- graphtool:page=2 -->\n\nTwo.\n"
    )
    assert [call[1] for call in llm.calls] == [
        PdfBatchConversion,
        PdfBatchConversion,
        PdfPageConversion,
        PdfPageConversion,
    ]
    second_page_context = llm.calls[3][0][1].content[0].text
    assert "Previous heading path: ['One']" in second_page_context
    assert "<!-- graphtool:page=1 -->\n\n# One" in second_page_context
    assert "Falling back to individual PDF pages" in caplog.text
    assert "Recovered PDF batch conversion with individual pages" in caplog.text


def test_convert_pdf_fails_when_individual_page_fallback_is_invalid(
    monkeypatch,
    tmp_path,
):
    _prepare(monkeypatch, ["One", "Two"])
    path = tmp_path / "manual.pdf"
    path.write_bytes(b"pdf")
    cache_dir = tmp_path / "cache"
    llm = FakeLLM(
        [
            _conversion((2, "Two.")),
            _conversion((2, "Two.")),
            PdfPageConversion(markdown=""),
        ]
    )

    with pytest.raises(
        ValueError,
        match="empty Markdown without marking the page blank",
    ):
        convert_pdf_to_markdown(
            path,
            "documents/manual.pdf",
            llm,
            cache_dir,
        )

    batch_path = cache_dir / source_key("documents/manual.pdf") / "batches"
    assert not list(batch_path.glob("*.json"))


def test_convert_pdf_discards_blank_page_markdown_and_continues(monkeypatch, tmp_path):
    _prepare(monkeypatch, ["Template footer", "Content"])
    path = tmp_path / "blank.pdf"
    path.write_bytes(b"pdf")
    llm = FakeLLM(
        [
            PdfBatchConversion(
                pages=[
                    ConvertedPdfPage(
                        page_number=1,
                        markdown="Confidentiality footer",
                        is_blank=True,
                    ),
                    ConvertedPdfPage(page_number=2, markdown="# Content"),
                ]
            )
        ]
    )

    markdown = convert_pdf_to_markdown(
        path,
        "documents/blank.pdf",
        llm,
        tmp_path / "cache",
    )

    assert markdown == (
        "<!-- graphtool:page=1 -->\n\n"
        "<!-- graphtool:page=2 -->\n\n# Content\n"
    )


def test_validate_conversion_rejects_empty_unmarked_page():
    with pytest.raises(
        ValueError,
        match="empty Markdown without marking the page blank",
    ):
        pdf_conversion.validate_conversion(
            _conversion((1, "")),
            [1],
            "documents/manual.pdf",
        )


def test_pdf_prompt_defines_template_only_pages_as_blank():
    assert "confidentiality labels" in pdf_prompts.BATCH_SYSTEM_PROMPT
    assert "set is_blank to true and return empty Markdown" in pdf_prompts.BATCH_SYSTEM_PROMPT
    assert "still return one page record for every requested page" in pdf_prompts.BATCH_SYSTEM_PROMPT


def test_convert_pdf_invalidates_cache_when_prompt_revision_changes(
    monkeypatch,
    tmp_path,
):
    render_calls = _prepare(monkeypatch, ["Text"])
    path = tmp_path / "manual.pdf"
    path.write_bytes(b"pdf")
    cache_dir = tmp_path / "cache"
    convert_pdf_to_markdown(
        path,
        "documents/manual.pdf",
        FakeLLM([_conversion((1, "First."))]),
        cache_dir,
    )
    monkeypatch.setattr(
        pdf_cache,
        "PDF_PROMPT_REVISION",
        pdf_cache.PDF_PROMPT_REVISION + 1,
    )

    markdown = convert_pdf_to_markdown(
        path,
        "documents/manual.pdf",
        FakeLLM([_conversion((1, "Second."))]),
        cache_dir,
    )

    assert render_calls == [[1], [1]]
    assert "Second." in markdown


def test_convert_pdf_fails_clearly_without_poppler(monkeypatch, tmp_path):
    monkeypatch.setattr(pdf, "extract_page_texts", lambda path, source: ["Text"])
    monkeypatch.setattr(pdf.shutil, "which", lambda name: None)
    path = tmp_path / "manual.pdf"
    path.write_bytes(b"pdf")

    with pytest.raises(RuntimeError, match="Poppler pdftoppm was not found"):
        convert_pdf_to_markdown(
            path,
            "documents/manual.pdf",
            FakeLLM([]),
            tmp_path / "cache",
        )


def test_convert_pdf_rejects_invalid_pdf(tmp_path):
    path = tmp_path / "invalid.pdf"
    path.write_bytes(b"not a pdf")

    with pytest.raises(ValueError, match="Cannot read PDF"):
        convert_pdf_to_markdown(
            path,
            "documents/invalid.pdf",
            FakeLLM([]),
            tmp_path / "cache",
        )


def test_convert_pdf_rejects_password_protected_pdf(tmp_path):
    path = tmp_path / "protected.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("secret")
    with path.open("wb") as file:
        writer.write(file)

    with pytest.raises(ValueError, match="Password-protected PDF"):
        convert_pdf_to_markdown(
            path,
            "documents/protected.pdf",
            FakeLLM([]),
            tmp_path / "cache",
        )


def test_completed_cache_manifest_records_fast_model_and_markdown_hash(
    monkeypatch,
    tmp_path,
):
    _prepare(monkeypatch, ["Text"])
    path = tmp_path / "manual.pdf"
    path.write_bytes(b"pdf")
    cache_dir = tmp_path / "cache"
    source = "documents/manual.pdf"

    convert_pdf_to_markdown(
        path,
        source,
        FakeLLM([_conversion((1, "Text."))]),
        cache_dir,
    )

    manifest_path = cache_dir / source_key(source) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["model"] == "fast-deployment"
    assert manifest["complete"] is True
    assert manifest["markdown_hash"]
    assert not list((cache_dir / source_key(source)).glob("*.tmp"))


def test_make_page_batches_respects_page_and_extracted_text_limits():
    page_texts = ["a", "b", "x" * 16_000, "c"]

    batches = pdf_pages.make_page_batches(page_texts)

    assert [[page_number for page_number, _ in batch] for batch in batches] == [
        [1, 2],
        [3],
        [4],
    ]


def test_render_pages_uses_poppler_and_removes_temporary_images(
    monkeypatch,
    tmp_path,
):
    temporary_paths = []
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        prefix = command[-1]
        first_page = int(command[command.index("-f") + 1])
        last_page = int(command[command.index("-l") + 1])
        for page_number in range(first_page, last_page + 1):
            image_path = pdf_pages.Path(f"{prefix}-{page_number}.png")
            temporary_paths.append(image_path)
            image_path.write_bytes(f"image-{page_number}".encode())

    monkeypatch.setattr(pdf_pages.subprocess, "run", fake_run)

    images = pdf_pages.render_pages(
        tmp_path / "manual.pdf",
        [3, 4],
        "/usr/bin/pdftoppm",
        "documents/manual.pdf",
    )

    assert images == [b"image-3", b"image-4"]
    assert commands[0][commands[0].index("-r") + 1] == "150"
    assert all(not path.exists() for path in temporary_paths)
