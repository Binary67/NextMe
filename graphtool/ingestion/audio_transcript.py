from pathlib import Path

from pydantic import BaseModel

from graphtool.ingestion.audio_media import chunk_boundaries


class AudioTranscriptChunk(BaseModel):
    index: int
    start_milliseconds: int
    end_milliseconds: int
    text: str


def normalize_transcript(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def validate_chunk(
    chunk: AudioTranscriptChunk,
    expected_index: int,
    expected_start: int,
    expected_end: int,
    source: str,
) -> AudioTranscriptChunk:
    if (
        chunk.index != expected_index
        or chunk.start_milliseconds != expected_start
        or chunk.end_milliseconds != expected_end
    ):
        raise ValueError(
            f"Cached audio transcription for {source!r} chunk {expected_index} "
            "has unexpected boundaries."
        )
    text = normalize_transcript(chunk.text)
    if not text:
        raise ValueError(
            f"Cached audio transcription for {source!r} chunk {expected_index} "
            "is empty."
        )
    return chunk.model_copy(update={"text": text})


def load_cached_chunks(
    source_cache_dir: Path,
    duration_milliseconds: int,
    chunk_count: int,
    source: str,
) -> list[AudioTranscriptChunk]:
    boundaries = chunk_boundaries(duration_milliseconds)
    if len(boundaries) != chunk_count:
        raise ValueError(
            f"Cached audio transcription for {source!r} has unexpected boundaries."
        )
    chunks = []
    for index, (start_milliseconds, end_milliseconds) in enumerate(boundaries):
        chunk_path = source_cache_dir / "chunks" / f"{index:05d}.json"
        if not chunk_path.exists():
            raise ValueError(
                f"Cached audio transcription for {source!r} is missing "
                f"chunk {index}."
            )
        chunk = AudioTranscriptChunk.model_validate_json(
            chunk_path.read_text(encoding="utf-8")
        )
        chunks.append(
            validate_chunk(
                chunk,
                index,
                start_milliseconds,
                end_milliseconds,
                source,
            )
        )
    return chunks
