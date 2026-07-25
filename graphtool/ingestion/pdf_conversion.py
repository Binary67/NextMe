import logging

from pydantic import BaseModel, Field

from graphtool.ingestion.pdf_prompts import (
    BATCH_SYSTEM_PROMPT,
    SINGLE_PAGE_SYSTEM_PROMPT,
)
from graphtool.llm.base import LLMClient
from graphtool.llm.types import LLMImageContent, LLMMessage, LLMTextContent
from graphtool.run_logging import LOGGER_NAME

PDF_CONTEXT_TAIL_CHARS = 1_000
_PAGE_MARKER_TEMPLATE = "<!-- graphtool:page={page_number} -->"
RUN_LOGGER = logging.getLogger(LOGGER_NAME)


class ConvertedPdfPage(BaseModel):
    page_number: int
    markdown: str
    is_blank: bool = False
    warnings: list[str] = Field(default_factory=list)


class PdfBatchConversion(BaseModel):
    pages: list[ConvertedPdfPage]
    ending_heading_path: list[str] = Field(default_factory=list)


class PdfPageConversion(BaseModel):
    markdown: str
    is_blank: bool = False
    warnings: list[str] = Field(default_factory=list)
    ending_heading_path: list[str] = Field(default_factory=list)


def convert_batch(
    pages: list[tuple[int, str]],
    images: list[bytes],
    heading_path: list[str],
    previous_markdown: str,
    llm: LLMClient,
    correction: str | None = None,
) -> PdfBatchConversion:
    context = (
        "Convert the requested pages below. Previous heading path: "
        f"{heading_path or ['None']}. The following previous Markdown is context "
        "only; do not repeat it:\n\n"
        f"{previous_markdown or '[None]'}"
    )
    if correction is not None:
        context = f"{context}\n\nCorrection required:\n{correction}"
    content = [LLMTextContent(text=context)]
    for (page_number, extracted_text), image in zip(pages, images, strict=True):
        content.extend(
            [
                LLMTextContent(
                    text=(
                        f"Page {page_number} extracted text (use as transcription "
                        f"grounding):\n\n{extracted_text or '[No extracted text]'}"
                    )
                ),
                LLMImageContent(data=image, detail="high"),
            ]
        )

    return llm.generate_structured(
        [
            LLMMessage(role="system", content=BATCH_SYSTEM_PROMPT),
            LLMMessage(role="user", content=tuple(content)),
        ],
        PdfBatchConversion,
    )


def convert_batch_with_recovery(
    pages: list[tuple[int, str]],
    images: list[bytes],
    heading_path: list[str],
    previous_markdown: str,
    llm: LLMClient,
    source: str,
) -> PdfBatchConversion:
    page_numbers = [page_number for page_number, _ in pages]
    conversion = convert_batch(
        pages,
        images,
        heading_path,
        previous_markdown,
        llm,
    )
    try:
        conversion = validate_conversion(conversion, page_numbers, source)
    except ValueError as first_error:
        RUN_LOGGER.warning(
            "Retrying PDF batch conversion source=%s pages=%s error=%s",
            source,
            page_numbers,
            first_error,
        )
        correction = (
            f"The previous response failed validation: {first_error} "
            "Return exactly one page record for each of these page numbers, in this "
            f"exact order: {page_numbers}. Never omit a page record; use is_blank=true "
            "and empty Markdown when a page has no meaningful content."
        )
        conversion = convert_batch(
            pages,
            images,
            heading_path,
            previous_markdown,
            llm,
            correction,
        )
        try:
            conversion = validate_conversion(conversion, page_numbers, source)
        except ValueError as retry_error:
            RUN_LOGGER.warning(
                "Falling back to individual PDF pages source=%s pages=%s error=%s",
                source,
                page_numbers,
                retry_error,
            )
            conversion = _convert_pages_individually(
                pages,
                images,
                heading_path,
                previous_markdown,
                llm,
                source,
            )
            RUN_LOGGER.info(
                "Recovered PDF batch conversion with individual pages source=%s "
                "pages=%s",
                source,
                page_numbers,
            )
        else:
            RUN_LOGGER.info(
                "Recovered PDF batch conversion on retry source=%s pages=%s",
                source,
                page_numbers,
            )
    return conversion


def _convert_pages_individually(
    pages: list[tuple[int, str]],
    images: list[bytes],
    heading_path: list[str],
    previous_markdown: str,
    llm: LLMClient,
    source: str,
) -> PdfBatchConversion:
    converted_pages = []
    current_heading_path = list(heading_path)
    markdown_tail = previous_markdown
    for (page_number, extracted_text), image in zip(pages, images, strict=True):
        page_conversion = _convert_page(
            page_number,
            extracted_text,
            image,
            current_heading_path,
            markdown_tail,
            llm,
        )
        conversion = validate_conversion(
            PdfBatchConversion(
                pages=[
                    ConvertedPdfPage(
                        page_number=page_number,
                        markdown=page_conversion.markdown,
                        is_blank=page_conversion.is_blank,
                        warnings=page_conversion.warnings,
                    )
                ],
                ending_heading_path=page_conversion.ending_heading_path,
            ),
            [page_number],
            source,
        )
        converted_pages.extend(conversion.pages)
        current_heading_path = conversion.ending_heading_path
        markdown_tail = extend_markdown_tail(markdown_tail, conversion.pages)
    return PdfBatchConversion(
        pages=converted_pages,
        ending_heading_path=current_heading_path,
    )


def _convert_page(
    page_number: int,
    extracted_text: str,
    image: bytes,
    heading_path: list[str],
    previous_markdown: str,
    llm: LLMClient,
) -> PdfPageConversion:
    context = (
        f"Convert page {page_number}. Previous heading path: "
        f"{heading_path or ['None']}. The following previous Markdown is context "
        "only; do not repeat it:\n\n"
        f"{previous_markdown or '[None]'}"
    )
    return llm.generate_structured(
        [
            LLMMessage(role="system", content=SINGLE_PAGE_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=(
                    LLMTextContent(text=context),
                    LLMTextContent(
                        text=(
                            f"Page {page_number} extracted text (use as transcription "
                            f"grounding):\n\n{extracted_text or '[No extracted text]'}"
                        )
                    ),
                    LLMImageContent(data=image, detail="high"),
                ),
            ),
        ],
        PdfPageConversion,
    )


def validate_conversion(
    conversion: PdfBatchConversion,
    expected_page_numbers: list[int],
    source: str,
) -> PdfBatchConversion:
    actual_page_numbers = [page.page_number for page in conversion.pages]
    if actual_page_numbers != expected_page_numbers:
        raise ValueError(
            f"PDF conversion for {source!r} expected pages "
            f"{expected_page_numbers}, received {actual_page_numbers}."
        )

    normalized_pages = []
    for page in conversion.pages:
        markdown = _normalize_markdown(page.markdown)
        if page.is_blank:
            markdown = ""
        elif not markdown:
            raise ValueError(
                f"PDF conversion for {source!r} page {page.page_number} returned "
                "empty Markdown without marking the page blank."
            )
        normalized_pages.append(page.model_copy(update={"markdown": markdown}))

    return conversion.model_copy(update={"pages": normalized_pages})


def assemble_markdown(pages: list[ConvertedPdfPage]) -> str:
    blocks = []
    for page in pages:
        marker = _PAGE_MARKER_TEMPLATE.format(page_number=page.page_number)
        blocks.append(f"{marker}\n\n{page.markdown}".rstrip())
    return "\n\n".join(blocks).rstrip() + "\n"


def extend_markdown_tail(
    markdown_tail: str,
    pages: list[ConvertedPdfPage],
) -> str:
    if markdown_tail:
        markdown_tail += "\n"
    return (markdown_tail + assemble_markdown(pages))[-PDF_CONTEXT_TAIL_CHARS:]


def _normalize_markdown(markdown: str) -> str:
    return markdown.replace("\r\n", "\n").replace("\r", "\n").strip()
