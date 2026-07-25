import subprocess
import tempfile
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

PDF_BATCH_MAX_PAGES = 8
PDF_BATCH_MAX_EXTRACTED_CHARS = 16_000
PDF_RENDER_DPI = 150


def page_unit(source: str, count: int) -> str:
    unit = "slide" if Path(source).suffix.lower() == ".pptx" else "page"
    return unit if count == 1 else f"{unit}s"


def page_range(page_numbers: list[int]) -> str:
    if len(page_numbers) == 1:
        return str(page_numbers[0])
    return f"{page_numbers[0]}-{page_numbers[-1]}"


def extract_page_texts(pdf_path: Path, source: str) -> list[str]:
    try:
        reader = PdfReader(pdf_path)
    except (OSError, PdfReadError) as exc:
        raise ValueError(f"Cannot read PDF {source!r}.") from exc

    if reader.is_encrypted:
        raise ValueError(f"Password-protected PDF {source!r} is not supported.")
    if not reader.pages:
        raise ValueError(f"PDF {source!r} has no pages.")

    texts = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            texts.append(page.extract_text() or "")
        except Exception as exc:
            raise ValueError(
                f"Cannot extract text from {source!r} page {page_number}."
            ) from exc
    return texts


def make_page_batches(page_texts: list[str]) -> list[list[tuple[int, str]]]:
    batches: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    current_chars = 0
    for page_number, text in enumerate(page_texts, start=1):
        if current and (
            len(current) == PDF_BATCH_MAX_PAGES
            or current_chars + len(text) > PDF_BATCH_MAX_EXTRACTED_CHARS
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append((page_number, text))
        current_chars += len(text)
    if current:
        batches.append(current)
    return batches


def render_pages(
    pdf_path: Path,
    page_numbers: list[int],
    pdftoppm: str,
    source: str,
) -> list[bytes]:
    with tempfile.TemporaryDirectory(prefix="graphtool-pdf-") as temporary_dir:
        prefix = Path(temporary_dir) / "page"
        command = [
            pdftoppm,
            "-f",
            str(page_numbers[0]),
            "-l",
            str(page_numbers[-1]),
            "-r",
            str(PDF_RENDER_DPI),
            "-png",
            str(pdf_path),
            str(prefix),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or "unknown Poppler error"
            raise RuntimeError(
                f"Cannot render {source!r} pages {page_numbers[0]}-"
                f"{page_numbers[-1]}: {detail}"
            ) from exc

        image_paths = sorted(Path(temporary_dir).glob("page-*.png"))
        if len(image_paths) != len(page_numbers):
            raise RuntimeError(
                f"Expected {len(page_numbers)} rendered pages for {source!r}, "
                f"received {len(image_paths)}."
            )
        return [image_path.read_bytes() for image_path in image_paths]
