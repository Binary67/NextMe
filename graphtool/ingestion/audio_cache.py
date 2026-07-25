from pathlib import Path

from pydantic import BaseModel

from graphtool.ingestion.audio_assembly import AUDIO_ASSEMBLY_REVISION
from graphtool.ingestion.audio_corrections import AUDIO_CORRECTION_REVISION
from graphtool.ingestion.audio_media import (
    AUDIO_BITRATE,
    AUDIO_CHUNK_MILLISECONDS,
    AUDIO_OVERLAP_MILLISECONDS,
    AUDIO_SAMPLE_RATE,
)

AUDIO_TRANSCRIPTION_FORMAT_REVISION = 3


class AudioConversionManifest(BaseModel):
    source_hash: str
    model: str
    glossary_hash: str
    correction_model: str
    correction_revision: int
    correction_input_hash: str
    corrections_hash: str
    format_revision: int
    assembly_revision: int
    chunk_milliseconds: int
    overlap_milliseconds: int
    sample_rate: int
    bitrate: str
    duration_milliseconds: int
    chunk_count: int
    complete: bool = False
    markdown_hash: str | None = None


def expected_manifest(
    source_hash: str,
    model: str,
    glossary_hash: str,
    correction_model: str,
    duration_milliseconds: int,
    chunk_count: int,
) -> AudioConversionManifest:
    return AudioConversionManifest(
        source_hash=source_hash,
        model=model,
        glossary_hash=glossary_hash,
        correction_model=correction_model,
        correction_revision=AUDIO_CORRECTION_REVISION,
        correction_input_hash="",
        corrections_hash="",
        format_revision=AUDIO_TRANSCRIPTION_FORMAT_REVISION,
        assembly_revision=AUDIO_ASSEMBLY_REVISION,
        chunk_milliseconds=AUDIO_CHUNK_MILLISECONDS,
        overlap_milliseconds=AUDIO_OVERLAP_MILLISECONDS,
        sample_rate=AUDIO_SAMPLE_RATE,
        bitrate=AUDIO_BITRATE,
        duration_milliseconds=duration_milliseconds,
        chunk_count=chunk_count,
    )


def load_manifest(path: Path) -> AudioConversionManifest | None:
    if not path.exists():
        return None
    return AudioConversionManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def same_conversion(
    current: AudioConversionManifest,
    expected: AudioConversionManifest,
) -> bool:
    fields = (
        "source_hash",
        "model",
        "glossary_hash",
        "format_revision",
        "chunk_milliseconds",
        "overlap_milliseconds",
        "sample_rate",
        "bitrate",
        "duration_milliseconds",
        "chunk_count",
    )
    return all(getattr(current, field) == getattr(expected, field) for field in fields)


def same_source_and_settings(
    manifest: AudioConversionManifest,
    source_hash: str,
    model: str,
    glossary_hash: str,
    correction_model: str,
) -> bool:
    return (
        manifest.source_hash == source_hash
        and manifest.model == model
        and manifest.glossary_hash == glossary_hash
        and manifest.correction_model == correction_model
        and manifest.correction_revision == AUDIO_CORRECTION_REVISION
        and manifest.format_revision == AUDIO_TRANSCRIPTION_FORMAT_REVISION
        and manifest.assembly_revision == AUDIO_ASSEMBLY_REVISION
        and manifest.chunk_milliseconds == AUDIO_CHUNK_MILLISECONDS
        and manifest.overlap_milliseconds == AUDIO_OVERLAP_MILLISECONDS
        and manifest.sample_rate == AUDIO_SAMPLE_RATE
        and manifest.bitrate == AUDIO_BITRATE
    )
