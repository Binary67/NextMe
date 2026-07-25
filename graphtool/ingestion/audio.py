import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

from graphtool.ingestion.audio_assembly import assemble_markdown
from graphtool.ingestion.audio_cache import (
    expected_manifest,
    load_manifest,
    same_conversion,
    same_source_and_settings,
)
from graphtool.ingestion.audio_corrections import (
    AUDIO_CORRECTION_REVISION,
    correction_input_hash,
    generate_corrections,
    load_corrections,
    merge_corrections,
    serialize_corrections,
)
from graphtool.ingestion.audio_glossary import context_prompt, glossary_prompt
from graphtool.ingestion.audio_media import (
    chunk_boundaries,
    probe_duration,
    render_chunk,
)
from graphtool.ingestion.audio_transcript import (
    AudioTranscriptChunk,
    load_cached_chunks,
    normalize_transcript,
    validate_chunk,
)
from graphtool.ingestion.cache_io import (
    file_hash,
    text_hash,
    write_model_atomic,
    write_text_atomic,
)
from graphtool.llm.base import AudioTranscriptionClient, LLMClient
from graphtool.source import source_key


def convert_audio_to_markdown(
    path: str | Path,
    source: str,
    transcriber: AudioTranscriptionClient,
    corrector: LLMClient,
    cache_dir: str | Path,
    terms: Sequence[str],
) -> str:
    audio_path = Path(path)
    source_hash = file_hash(audio_path)
    prompt_prefix = glossary_prompt(terms)
    glossary_hash = text_hash(prompt_prefix or "")

    source_cache_dir = Path(cache_dir) / source_key(source)
    manifest_path = source_cache_dir / "manifest.json"
    markdown_path = source_cache_dir / "document.md"
    corrections_path = source_cache_dir / "corrections.jsonl"
    corrections, corrections_content = load_corrections(corrections_path)
    manifest = load_manifest(manifest_path)
    if manifest is not None and same_source_and_settings(
        manifest,
        source_hash,
        transcriber.transcription_model,
        glossary_hash,
        corrector.text_model,
    ):
        if manifest.complete and markdown_path.exists():
            markdown = markdown_path.read_text(encoding="utf-8")
            corrections_hash = text_hash(corrections_content)
            if (
                text_hash(markdown) == manifest.markdown_hash
                and corrections_hash == manifest.corrections_hash
            ):
                return markdown
            chunks = load_cached_chunks(
                source_cache_dir,
                manifest.duration_milliseconds,
                manifest.chunk_count,
                source,
            )
            markdown = assemble_markdown(
                audio_path.name,
                source,
                chunks,
                corrections,
                terms,
            )
            write_text_atomic(markdown_path, markdown)
            completed_manifest = manifest.model_copy(
                update={
                    "corrections_hash": corrections_hash,
                    "markdown_hash": text_hash(markdown),
                }
            )
            write_model_atomic(manifest_path, completed_manifest)
            return markdown

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError(f"Cannot transcribe {source!r}: ffprobe was not found.")
    duration_milliseconds = probe_duration(audio_path, source, ffprobe)
    boundaries = chunk_boundaries(duration_milliseconds)
    manifest_for_run = expected_manifest(
        source_hash,
        transcriber.transcription_model,
        glossary_hash,
        corrector.text_model,
        duration_milliseconds,
        len(boundaries),
    )
    if manifest is None or not same_conversion(manifest, manifest_for_run):
        if source_cache_dir.exists():
            shutil.rmtree(source_cache_dir)
        manifest = None

    source_cache_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = source_cache_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    if manifest is None:
        write_model_atomic(manifest_path, manifest_for_run)

    ffmpeg: str | None = None
    chunks: list[AudioTranscriptChunk] = []
    with tempfile.TemporaryDirectory(prefix="graphtool-audio-") as temporary_dir:
        for index, (start_milliseconds, end_milliseconds) in enumerate(boundaries):
            chunk_cache_path = chunks_dir / f"{index:05d}.json"
            if chunk_cache_path.exists():
                chunk = AudioTranscriptChunk.model_validate_json(
                    chunk_cache_path.read_text(encoding="utf-8")
                )
                chunk = validate_chunk(
                    chunk,
                    index,
                    start_milliseconds,
                    end_milliseconds,
                    source,
                )
            else:
                if ffmpeg is None:
                    ffmpeg = shutil.which("ffmpeg")
                    if ffmpeg is None:
                        raise RuntimeError(
                            f"Cannot transcribe {source!r}: ffmpeg was not found."
                        )
                chunk_path = Path(temporary_dir) / f"{index:05d}.mp3"
                render_chunk(
                    audio_path,
                    chunk_path,
                    start_milliseconds,
                    end_milliseconds,
                    ffmpeg,
                    source,
                )
                prompt = context_prompt(
                    prompt_prefix,
                    chunks[-1].text if chunks else None,
                )
                text = normalize_transcript(
                    transcriber.transcribe_audio(chunk_path, prompt=prompt)
                )
                if not text:
                    raise ValueError(
                        f"Audio transcription for {source!r} chunk {index} was empty."
                    )
                chunk = AudioTranscriptChunk(
                    index=index,
                    start_milliseconds=start_milliseconds,
                    end_milliseconds=end_milliseconds,
                    text=text,
                )
                write_model_atomic(chunk_cache_path, chunk)
            chunks.append(chunk)

    current_correction_input_hash = correction_input_hash(chunks, terms, corrector)
    corrections_are_stale = (
        manifest is None
        or manifest.correction_input_hash != current_correction_input_hash
        or manifest.correction_model != corrector.text_model
        or manifest.correction_revision != AUDIO_CORRECTION_REVISION
    )
    if corrections_are_stale:
        corrections = merge_corrections(
            generate_corrections(source, chunks, terms, corrector),
            corrections,
            source,
            chunks,
            terms,
        )
        corrections_content = serialize_corrections(corrections)
        write_text_atomic(corrections_path, corrections_content)

    markdown = assemble_markdown(
        audio_path.name,
        source,
        chunks,
        corrections,
        terms,
    )
    write_text_atomic(markdown_path, markdown)
    completed_manifest = manifest_for_run.model_copy(
        update={
            "complete": True,
            "correction_input_hash": current_correction_input_hash,
            "corrections_hash": text_hash(corrections_content),
            "markdown_hash": text_hash(markdown),
        }
    )
    write_model_atomic(manifest_path, completed_manifest)
    return markdown
