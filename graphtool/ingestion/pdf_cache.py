from pathlib import Path

from pydantic import BaseModel

from graphtool.ingestion.pdf_pages import (
    PDF_BATCH_MAX_EXTRACTED_CHARS,
    PDF_BATCH_MAX_PAGES,
    PDF_RENDER_DPI,
)

PDF_PROMPT_REVISION = 3


class PdfConversionManifest(BaseModel):
    source_hash: str
    model: str
    prompt_revision: int
    render_dpi: int
    batch_max_pages: int
    batch_max_extracted_chars: int
    page_count: int
    complete: bool = False
    markdown_hash: str | None = None


def expected_manifest(
    source_hash: str,
    model: str,
    page_count: int,
) -> PdfConversionManifest:
    return PdfConversionManifest(
        source_hash=source_hash,
        model=model,
        prompt_revision=PDF_PROMPT_REVISION,
        render_dpi=PDF_RENDER_DPI,
        batch_max_pages=PDF_BATCH_MAX_PAGES,
        batch_max_extracted_chars=PDF_BATCH_MAX_EXTRACTED_CHARS,
        page_count=page_count,
    )


def load_manifest(path: Path) -> PdfConversionManifest | None:
    if not path.exists():
        return None
    return PdfConversionManifest.model_validate_json(path.read_text())


def same_conversion(
    current: PdfConversionManifest,
    expected: PdfConversionManifest,
) -> bool:
    fields = (
        "source_hash",
        "model",
        "prompt_revision",
        "render_dpi",
        "batch_max_pages",
        "batch_max_extracted_chars",
        "page_count",
    )
    return all(getattr(current, field) == getattr(expected, field) for field in fields)


def same_source_and_settings(
    manifest: PdfConversionManifest,
    source_hash: str,
    model: str,
) -> bool:
    return (
        manifest.source_hash == source_hash
        and manifest.model == model
        and manifest.prompt_revision == PDF_PROMPT_REVISION
        and manifest.render_dpi == PDF_RENDER_DPI
        and manifest.batch_max_pages == PDF_BATCH_MAX_PAGES
        and manifest.batch_max_extracted_chars == PDF_BATCH_MAX_EXTRACTED_CHARS
    )


def batch_file_name(page_numbers: list[int]) -> str:
    return f"pages-{page_numbers[0]:05d}-{page_numbers[-1]:05d}.json"
