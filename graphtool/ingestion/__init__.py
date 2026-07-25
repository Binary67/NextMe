"""Document discovery and normalization."""

from graphtool.ingestion.audio import convert_audio_to_markdown
from graphtool.ingestion.audio_glossary import load_audio_transcription_terms
from graphtool.ingestion.documents import load_documents
from graphtool.ingestion.pdf import convert_pdf_to_markdown
from graphtool.ingestion.presentation import (
    convert_pptx_to_markdown,
    convert_pptx_to_pdf,
)

__all__ = [
    "convert_audio_to_markdown",
    "load_audio_transcription_terms",
    "convert_pdf_to_markdown",
    "convert_pptx_to_markdown",
    "convert_pptx_to_pdf",
    "load_documents",
]
