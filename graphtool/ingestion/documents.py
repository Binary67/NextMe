import logging
from collections.abc import Sequence
from pathlib import Path

from graphtool.ingestion.audio import convert_audio_to_markdown
from graphtool.ingestion.pdf import convert_pdf_to_markdown
from graphtool.ingestion.presentation import convert_pptx_to_markdown
from graphtool.llm.base import AudioTranscriptionClient, LLMClient
from graphtool.run_logging import LOGGER_NAME

_AUDIO_SUFFIXES = {
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".ogg",
    ".wav",
    ".webm",
}
_SUPPORTED_SUFFIXES = {".md", ".pdf", ".pptx", *_AUDIO_SUFFIXES}
RUN_LOGGER = logging.getLogger(LOGGER_NAME)


def load_documents(
    directory: str | Path,
    *,
    source_root: str | Path,
    pdf_llm: LLMClient,
    pdf_cache_dir: str | Path,
    presentation_cache_dir: str | Path,
    audio_transcriber: AudioTranscriptionClient,
    audio_corrector: LLMClient,
    audio_cache_dir: str | Path,
    audio_transcription_terms: Sequence[str],
) -> dict[str, str]:
    path = Path(directory)
    if not path.exists():
        return {}

    root = Path(source_root)
    documents = {}
    document_paths = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in _SUPPORTED_SUFFIXES
    )
    counts = (
        (sum(path.suffix.lower() == ".md" for path in document_paths), "Markdown"),
        (sum(path.suffix.lower() == ".pdf" for path in document_paths), "PDF"),
        (
            sum(path.suffix.lower() == ".pptx" for path in document_paths),
            "PowerPoint",
        ),
        (
            sum(path.suffix.lower() in _AUDIO_SUFFIXES for path in document_paths),
            "audio",
        ),
    )
    count_summary = ", ".join(
        f"{count} {label}" for count, label in counts if count
    )
    RUN_LOGGER.info(
        "Found %s supported %s%s",
        len(document_paths),
        "document" if len(document_paths) == 1 else "documents",
        f": {count_summary}" if count_summary else "",
    )

    for index, document_path in enumerate(document_paths, start=1):
        source = document_path.relative_to(root).as_posix()
        suffix = document_path.suffix.lower()
        if suffix == ".md":
            RUN_LOGGER.info(
                "[%s/%s] Reading Markdown: %s",
                index,
                len(document_paths),
                source,
            )
            documents[source] = document_path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            RUN_LOGGER.info(
                "[%s/%s] Processing PDF: %s",
                index,
                len(document_paths),
                source,
            )
            documents[source] = convert_pdf_to_markdown(
                document_path,
                source,
                pdf_llm,
                pdf_cache_dir,
            )
        elif suffix == ".pptx":
            RUN_LOGGER.info(
                "[%s/%s] Processing PowerPoint: %s",
                index,
                len(document_paths),
                source,
            )
            documents[source] = convert_pptx_to_markdown(
                document_path,
                source,
                pdf_llm,
                presentation_cache_dir,
                pdf_cache_dir,
            )
        else:
            RUN_LOGGER.info(
                "[%s/%s] Processing audio: %s",
                index,
                len(document_paths),
                source,
            )
            documents[source] = convert_audio_to_markdown(
                document_path,
                source,
                audio_transcriber,
                audio_corrector,
                audio_cache_dir,
                audio_transcription_terms,
            )
    return documents
