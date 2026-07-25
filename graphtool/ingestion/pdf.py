import logging
import shutil
from pathlib import Path

from graphtool.ingestion.cache_io import (
    file_hash,
    text_hash,
    write_model_atomic,
    write_text_atomic,
)
from graphtool.ingestion.pdf_cache import (
    batch_file_name,
    expected_manifest,
    load_manifest,
    same_conversion,
    same_source_and_settings,
)
from graphtool.ingestion.pdf_conversion import (
    PdfBatchConversion,
    assemble_markdown,
    convert_batch_with_recovery,
    extend_markdown_tail,
    validate_conversion,
)
from graphtool.ingestion.pdf_pages import (
    extract_page_texts,
    make_page_batches,
    page_range,
    page_unit,
    render_pages,
)
from graphtool.llm.base import LLMClient
from graphtool.run_logging import LOGGER_NAME
from graphtool.source import source_key

RUN_LOGGER = logging.getLogger(LOGGER_NAME)


def convert_pdf_to_markdown(
    path: str | Path,
    source: str,
    llm: LLMClient,
    cache_dir: str | Path,
) -> str:
    pdf_path = Path(path)
    source_hash = file_hash(pdf_path)
    source_cache_dir = Path(cache_dir) / source_key(source)
    manifest_path = source_cache_dir / "manifest.json"
    markdown_path = source_cache_dir / "document.md"
    manifest = load_manifest(manifest_path)
    if manifest is not None and same_source_and_settings(
        manifest,
        source_hash,
        llm.text_model,
    ):
        if manifest.complete and markdown_path.exists():
            markdown = markdown_path.read_text(encoding="utf-8")
            if text_hash(markdown) == manifest.markdown_hash:
                RUN_LOGGER.info(
                    "Using cached content conversion for %s (%s %s)",
                    source,
                    manifest.page_count,
                    page_unit(source, manifest.page_count),
                )
                return markdown

    page_texts = extract_page_texts(pdf_path, source)
    RUN_LOGGER.info(
        "Converting content for %s (%s %s)",
        source,
        len(page_texts),
        page_unit(source, len(page_texts)),
    )
    manifest_for_run = expected_manifest(
        source_hash,
        llm.text_model,
        len(page_texts),
    )
    if manifest is None or not same_conversion(manifest, manifest_for_run):
        if source_cache_dir.exists():
            shutil.rmtree(source_cache_dir)
        manifest = None

    source_cache_dir.mkdir(parents=True, exist_ok=True)
    batches_dir = source_cache_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    if manifest is None:
        write_model_atomic(manifest_path, manifest_for_run)

    converted_pages = []
    heading_path: list[str] = []
    markdown_tail = ""
    pdftoppm: str | None = None
    for page_batch in make_page_batches(page_texts):
        page_numbers = [page_number for page_number, _ in page_batch]
        batch_path = batches_dir / batch_file_name(page_numbers)
        if batch_path.exists():
            conversion = PdfBatchConversion.model_validate_json(batch_path.read_text())
            conversion = validate_conversion(conversion, page_numbers, source)
            RUN_LOGGER.debug(
                "Using cached content batch source=%s pages=%s",
                source,
                page_numbers,
            )
        else:
            if pdftoppm is None:
                pdftoppm = shutil.which("pdftoppm")
                if pdftoppm is None:
                    raise RuntimeError(
                        f"Cannot convert {source!r}: Poppler pdftoppm was not found."
                    )
            page_images = render_pages(
                pdf_path,
                page_numbers,
                pdftoppm,
                source,
            )
            conversion = convert_batch_with_recovery(
                page_batch,
                page_images,
                heading_path,
                markdown_tail,
                llm,
                source,
            )
            write_model_atomic(batch_path, conversion)
            RUN_LOGGER.info(
                "Processed %s: %s %s of %s",
                source,
                page_unit(source, len(page_numbers)),
                page_range(page_numbers),
                len(page_texts),
            )

        converted_pages.extend(conversion.pages)
        heading_path = conversion.ending_heading_path
        markdown_tail = extend_markdown_tail(markdown_tail, conversion.pages)

    markdown = assemble_markdown(converted_pages)
    write_text_atomic(markdown_path, markdown)
    completed_manifest = manifest_for_run.model_copy(
        update={"complete": True, "markdown_hash": text_hash(markdown)}
    )
    write_model_atomic(manifest_path, completed_manifest)
    RUN_LOGGER.info("Finished content conversion for %s", source)
    return markdown
