import hashlib
import math
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from graphtool.llm.base import AudioTranscriptionClient, LLMClient
from graphtool.llm.types import LLMMessage
from graphtool.source import source_key

AUDIO_CHUNK_MILLISECONDS = 20 * 60 * 1000
AUDIO_OVERLAP_MILLISECONDS = 5 * 1000
AUDIO_SAMPLE_RATE = 16_000
AUDIO_BITRATE = "64k"
AUDIO_MAX_CHUNK_BYTES = 24_000_000
AUDIO_CONTEXT_TAIL_CHARS = 500
AUDIO_TRANSCRIPTION_FORMAT_REVISION = 3
AUDIO_ASSEMBLY_REVISION = 1
AUDIO_CORRECTION_REVISION = 1
_MIN_OVERLAP_MATCH_TOKENS = 5
_MAX_OVERLAP_WINDOW_TOKENS = 100
_MIN_OVERLAP_SIMILARITY = 0.65


class AudioTranscriptChunk(BaseModel):
    index: int
    start_milliseconds: int
    end_milliseconds: int
    text: str


class _AudioTranscriptionGlossary(BaseModel):
    terms: list[str]


class _AudioTranscriptCorrectionProposal(BaseModel):
    original: str
    replacement: str
    context: str
    decision: Literal["apply", "review"]


class _AudioTranscriptCorrectionProposals(BaseModel):
    corrections: list[_AudioTranscriptCorrectionProposal]


class AudioTranscriptCorrection(BaseModel):
    id: str
    source: str
    chunk: int
    original: str
    replacement: str
    context: str
    decision: Literal["apply", "review", "reject"]
    reviewed: bool = False


class _AudioConversionManifest(BaseModel):
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


def convert_audio_to_markdown(
    path: str | Path,
    source: str,
    transcriber: AudioTranscriptionClient,
    corrector: LLMClient,
    cache_dir: str | Path,
    terms: Sequence[str],
) -> str:
    audio_path = Path(path)
    with audio_path.open("rb") as audio_file:
        source_hash = hashlib.file_digest(audio_file, "sha256").hexdigest()
    glossary_prompt = _glossary_prompt(terms)
    glossary_hash = _text_hash(glossary_prompt or "")

    source_cache_dir = Path(cache_dir) / source_key(source)
    manifest_path = source_cache_dir / "manifest.json"
    markdown_path = source_cache_dir / "document.md"
    corrections_path = source_cache_dir / "corrections.jsonl"
    corrections, corrections_content = _load_corrections(corrections_path)
    manifest = _load_manifest(manifest_path)
    if manifest is not None and _same_source_and_settings(
        manifest,
        source_hash,
        transcriber.transcription_model,
        glossary_hash,
        corrector.text_model,
    ):
        if manifest.complete and markdown_path.exists():
            markdown = markdown_path.read_text(encoding="utf-8")
            corrections_hash = _text_hash(corrections_content)
            if (
                _text_hash(markdown) == manifest.markdown_hash
                and corrections_hash == manifest.corrections_hash
            ):
                return markdown
            chunks = _load_cached_chunks(source_cache_dir, manifest, source)
            markdown = _assemble_markdown(
                audio_path.name,
                source,
                chunks,
                corrections,
                terms,
            )
            _write_text_atomic(markdown_path, markdown)
            completed_manifest = manifest.model_copy(
                update={
                    "corrections_hash": corrections_hash,
                    "markdown_hash": _text_hash(markdown),
                }
            )
            _write_model_atomic(manifest_path, completed_manifest)
            return markdown

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError(f"Cannot transcribe {source!r}: ffprobe was not found.")
    duration_milliseconds = _probe_duration(audio_path, source, ffprobe)
    boundaries = _chunk_boundaries(duration_milliseconds)
    expected_manifest = _AudioConversionManifest(
        source_hash=source_hash,
        model=transcriber.transcription_model,
        glossary_hash=glossary_hash,
        correction_model=corrector.text_model,
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
        chunk_count=len(boundaries),
    )
    if manifest is None or not _same_conversion(manifest, expected_manifest):
        if source_cache_dir.exists():
            shutil.rmtree(source_cache_dir)
        manifest = None

    source_cache_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = source_cache_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    if manifest is None:
        _write_model_atomic(manifest_path, expected_manifest)

    ffmpeg: str | None = None
    chunks: list[AudioTranscriptChunk] = []
    with tempfile.TemporaryDirectory(prefix="graphtool-audio-") as temporary_dir:
        for index, (start_milliseconds, end_milliseconds) in enumerate(boundaries):
            chunk_cache_path = chunks_dir / f"{index:05d}.json"
            if chunk_cache_path.exists():
                chunk = AudioTranscriptChunk.model_validate_json(
                    chunk_cache_path.read_text(encoding="utf-8")
                )
                chunk = _validate_chunk(
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
                _render_chunk(
                    audio_path,
                    chunk_path,
                    start_milliseconds,
                    end_milliseconds,
                    ffmpeg,
                    source,
                )
                prompt = _context_prompt(
                    glossary_prompt,
                    chunks[-1].text if chunks else None,
                )
                text = _normalize_transcript(
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
                _write_model_atomic(chunk_cache_path, chunk)
            chunks.append(chunk)

    correction_input_hash = _correction_input_hash(chunks, terms, corrector)
    corrections_are_stale = (
        manifest is None
        or manifest.correction_input_hash != correction_input_hash
        or manifest.correction_model != corrector.text_model
        or manifest.correction_revision != AUDIO_CORRECTION_REVISION
    )
    if corrections_are_stale:
        generated_corrections = _generate_corrections(
            source,
            chunks,
            terms,
            corrector,
        )
        corrections = _merge_corrections(
            generated_corrections,
            corrections,
            source,
            chunks,
            terms,
        )
        corrections_content = _serialize_corrections(corrections)
        _write_text_atomic(corrections_path, corrections_content)

    markdown = _assemble_markdown(
        audio_path.name,
        source,
        chunks,
        corrections,
        terms,
    )
    _write_text_atomic(markdown_path, markdown)
    completed_manifest = expected_manifest.model_copy(
        update={
            "complete": True,
            "correction_input_hash": correction_input_hash,
            "corrections_hash": _text_hash(corrections_content),
            "markdown_hash": _text_hash(markdown),
        }
    )
    _write_model_atomic(manifest_path, completed_manifest)
    return markdown


def load_audio_transcription_terms(path: str | Path) -> list[str]:
    glossary_path = Path(path)
    if not glossary_path.exists():
        return []
    glossary = _AudioTranscriptionGlossary.model_validate_json(
        glossary_path.read_text(encoding="utf-8")
    )
    terms = [term.strip() for term in glossary.terms]
    if any(not term for term in terms):
        raise ValueError(
            f"Audio transcription glossary {str(glossary_path)!r} "
            "contains an empty term."
        )
    return terms


def _load_corrections(
    path: Path,
) -> tuple[list[AudioTranscriptCorrection], str]:
    if not path.exists():
        return [], ""
    content = path.read_text(encoding="utf-8")
    corrections = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            correction = AudioTranscriptCorrection.model_validate_json(line)
        except ValueError as exc:
            raise ValueError(
                f"Audio correction ledger {str(path)!r} has an invalid "
                f"record on line {line_number}."
            ) from exc
        corrections.append(correction)
    return corrections, content


def _serialize_corrections(
    corrections: Sequence[AudioTranscriptCorrection],
) -> str:
    ordered = sorted(corrections, key=lambda item: (item.chunk, item.id))
    if not ordered:
        return ""
    return "\n".join(item.model_dump_json() for item in ordered) + "\n"


def _correction_input_hash(
    chunks: Sequence[AudioTranscriptChunk],
    terms: Sequence[str],
    corrector: LLMClient,
) -> str:
    values = [
        corrector.text_model,
        str(AUDIO_CORRECTION_REVISION),
        *terms,
        *(f"{chunk.index}:{_text_hash(chunk.text)}" for chunk in chunks),
    ]
    return _text_hash("\0".join(values))


def _generate_corrections(
    source: str,
    chunks: Sequence[AudioTranscriptChunk],
    terms: Sequence[str],
    corrector: LLMClient,
) -> list[AudioTranscriptCorrection]:
    if not terms:
        return []
    term_list = "\n".join(f"- {term}" for term in terms)
    corrections = []
    for chunk in chunks:
        proposals = corrector.generate_structured(
            (
                LLMMessage(
                    role="system",
                    content=(
                        "Find only proper-noun spelling discrepancies in an "
                        "audio transcript. Propose exact substring replacements "
                        "using the canonical terms supplied by the user. Do not "
                        "change punctuation, grammar, numbers, or wording. Set "
                        "decision to apply only when the intended canonical term "
                        "is unambiguous; otherwise set it to review. Context must "
                        "be the shortest exact excerpt from the transcript that "
                        "contains one occurrence of original and uniquely "
                        "identifies it."
                    ),
                ),
                LLMMessage(
                    role="user",
                    content=(
                        f"Canonical terms:\n{term_list}\n\n"
                        f"Transcript:\n{chunk.text}"
                    ),
                ),
            ),
            _AudioTranscriptCorrectionProposals,
        )
        chunk_corrections = []
        for proposal in proposals.corrections:
            original = proposal.original.strip()
            replacement = proposal.replacement.strip()
            context = proposal.context.strip()
            if (
                not original
                or not context
                or original == replacement
                or replacement not in terms
            ):
                continue
            correction = AudioTranscriptCorrection(
                id=_correction_id(
                    source,
                    chunk.index,
                    original,
                    replacement,
                    context,
                ),
                source=source,
                chunk=chunk.index,
                original=original,
                replacement=replacement,
                context=context,
                decision=proposal.decision,
            )
            if _locate_correction(chunk.text, correction) is not None:
                chunk_corrections.append(correction)
        corrections.extend(
            _mark_overlapping_corrections_for_review(
                chunk.text,
                chunk_corrections,
            )
        )
    return corrections


def _correction_id(
    source: str,
    chunk: int,
    original: str,
    replacement: str,
    context: str,
) -> str:
    value = "\0".join((source, str(chunk), original, replacement, context))
    return _text_hash(value)[:16]


def _locate_correction(
    text: str,
    correction: AudioTranscriptCorrection,
) -> tuple[int, int] | None:
    if text.count(correction.context) != 1:
        return None
    if correction.context.count(correction.original) != 1:
        return None
    context_start = text.index(correction.context)
    start = context_start + correction.context.index(correction.original)
    return start, start + len(correction.original)


def _mark_overlapping_corrections_for_review(
    text: str,
    corrections: Sequence[AudioTranscriptCorrection],
) -> list[AudioTranscriptCorrection]:
    located = []
    for correction in corrections:
        text_range = _locate_correction(text, correction)
        assert text_range is not None
        located.append((text_range, correction))
    overlapping_ids = _overlapping_correction_ids(located)
    return [
        correction.model_copy(update={"decision": "review"})
        if correction.id in overlapping_ids
        else correction
        for correction in corrections
    ]


def _merge_corrections(
    generated: Sequence[AudioTranscriptCorrection],
    existing: Sequence[AudioTranscriptCorrection],
    source: str,
    chunks: Sequence[AudioTranscriptChunk],
    terms: Sequence[str],
) -> list[AudioTranscriptCorrection]:
    chunk_text = {chunk.index: chunk.text for chunk in chunks}
    reviewed = {}
    for correction in existing:
        if (
            not correction.reviewed
            or correction.source != source
            or correction.replacement not in terms
        ):
            continue
        text = chunk_text.get(correction.chunk)
        if text is None or _locate_correction(text, correction) is None:
            continue
        reviewed[correction.id] = correction

    merged = {
        correction.id: reviewed.get(correction.id, correction)
        for correction in generated
    }
    for correction_id, correction in reviewed.items():
        merged.setdefault(correction_id, correction)
    return list(merged.values())


def _overlapping_correction_ids(
    located: Sequence[tuple[tuple[int, int], AudioTranscriptCorrection]],
) -> set[str]:
    ordered = sorted(located, key=lambda item: item[0])
    overlapping_ids = set()
    for index, (current_range, current) in enumerate(ordered):
        for other_range, other in ordered[index + 1 :]:
            if other_range[0] >= current_range[1]:
                break
            overlapping_ids.update((current.id, other.id))
    return overlapping_ids


def _probe_duration(audio_path: Path, source: str, ffprobe: str) -> int:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        duration = float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as exc:
        detail = getattr(exc, "stderr", "").strip() or "invalid duration"
        raise ValueError(f"Cannot read audio {source!r}: {detail}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Cannot read audio {source!r}: invalid duration")
    return math.ceil(duration * 1000)


def _chunk_boundaries(duration_milliseconds: int) -> list[tuple[int, int]]:
    boundaries = []
    start = 0
    while start < duration_milliseconds:
        end = min(
            start + AUDIO_CHUNK_MILLISECONDS + AUDIO_OVERLAP_MILLISECONDS,
            duration_milliseconds,
        )
        boundaries.append((start, end))
        if end == duration_milliseconds:
            break
        start += AUDIO_CHUNK_MILLISECONDS
    return boundaries


def _render_chunk(
    source_path: Path,
    chunk_path: Path,
    start_milliseconds: int,
    end_milliseconds: int,
    ffmpeg: str,
    source: str,
) -> None:
    command = [
        ffmpeg,
        "-v",
        "error",
        "-y",
        "-ss",
        _ffmpeg_time(start_milliseconds),
        "-i",
        str(source_path),
        "-t",
        _ffmpeg_time(end_milliseconds - start_milliseconds),
        "-vn",
        "-map_metadata",
        "-1",
        "-ac",
        "1",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-b:a",
        AUDIO_BITRATE,
        str(chunk_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"Cannot prepare audio {source!r}: {detail}") from exc
    if not chunk_path.exists() or chunk_path.stat().st_size == 0:
        raise RuntimeError(f"Cannot prepare audio {source!r}: ffmpeg produced no audio.")
    if chunk_path.stat().st_size > AUDIO_MAX_CHUNK_BYTES:
        raise ValueError(
            f"Prepared audio chunk for {source!r} exceeds "
            f"{AUDIO_MAX_CHUNK_BYTES} bytes."
        )


def _glossary_prompt(terms: Sequence[str]) -> str | None:
    if not terms:
        return None
    term_list = "\n".join(f"- {term}" for term in terms)
    return (
        "Expected proper nouns and exact spellings:\n"
        f"{term_list}\n"
        "Use these spellings only when they match the spoken audio."
    )


def _context_prompt(
    glossary_prompt: str | None,
    previous_text: str | None,
) -> str | None:
    previous_context = (
        previous_text[-AUDIO_CONTEXT_TAIL_CHARS:]
        if previous_text is not None
        else None
    )
    if glossary_prompt is None:
        return previous_context
    if previous_context is None:
        return glossary_prompt
    return f"{glossary_prompt}\n\nPrevious transcript context:\n{previous_context}"


def _load_cached_chunks(
    source_cache_dir: Path,
    manifest: _AudioConversionManifest,
    source: str,
) -> list[AudioTranscriptChunk]:
    boundaries = _chunk_boundaries(manifest.duration_milliseconds)
    if len(boundaries) != manifest.chunk_count:
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
            _validate_chunk(
                chunk,
                index,
                start_milliseconds,
                end_milliseconds,
                source,
            )
        )
    return chunks


def _assemble_markdown(
    file_name: str,
    source: str,
    chunks: list[AudioTranscriptChunk],
    corrections: Sequence[AudioTranscriptCorrection],
    terms: Sequence[str],
) -> str:
    blocks = [f"# Transcript: {file_name}"]
    previous_text = ""
    for chunk in chunks:
        corrected_text = _apply_corrections(
            source,
            chunk,
            corrections,
            terms,
        )
        text = _remove_fuzzy_overlap(previous_text, corrected_text)
        if text:
            blocks.append(
                f"## {_format_timestamp(chunk.start_milliseconds)}\n\n{text}"
            )
        previous_text = corrected_text
    return "\n\n".join(blocks).rstrip() + "\n"


def _apply_corrections(
    source: str,
    chunk: AudioTranscriptChunk,
    corrections: Sequence[AudioTranscriptCorrection],
    terms: Sequence[str],
) -> str:
    located = []
    for correction in corrections:
        if (
            correction.source != source
            or correction.chunk != chunk.index
            or correction.decision != "apply"
            or correction.replacement not in terms
        ):
            continue
        text_range = _locate_correction(chunk.text, correction)
        if text_range is not None:
            located.append((text_range, correction))

    overlapping_ids = _overlapping_correction_ids(located)

    text = chunk.text
    ordered = sorted(located, key=lambda item: item[0])
    for (start, end), correction in reversed(ordered):
        if correction.id not in overlapping_ids:
            text = f"{text[:start]}{correction.replacement}{text[end:]}"
    return text


def _remove_fuzzy_overlap(previous: str, current: str) -> str:
    if not previous:
        return current.strip()
    previous_tokens = _normalized_tokens(previous)[-_MAX_OVERLAP_WINDOW_TOKENS:]
    current_tokens = _normalized_tokens(current)[:_MAX_OVERLAP_WINDOW_TOKENS]
    if not previous_tokens or not current_tokens:
        return current.strip()

    previous_values = [token[0] for token in previous_tokens]
    current_values = [token[0] for token in current_tokens]
    previous_count = len(previous_values)
    current_count = len(current_values)
    # Each cell stores edits, exact matches, and the previous suffix start.
    scores: list[list[tuple[int, int, int]]] = [
        [(0, 0, 0)] * (current_count + 1)
        for _ in range(previous_count + 1)
    ]
    for previous_index in range(previous_count + 1):
        scores[previous_index][0] = (0, 0, previous_index)
    for current_index in range(1, current_count + 1):
        scores[0][current_index] = (current_index, 0, 0)

    for previous_index in range(1, previous_count + 1):
        for current_index in range(1, current_count + 1):
            is_match = (
                previous_values[previous_index - 1]
                == current_values[current_index - 1]
            )
            diagonal = scores[previous_index - 1][current_index - 1]
            delete = scores[previous_index - 1][current_index]
            insert = scores[previous_index][current_index - 1]
            candidates = (
                (
                    diagonal[0] + (0 if is_match else 1),
                    diagonal[1] + int(is_match),
                    diagonal[2],
                ),
                (delete[0] + 1, delete[1], delete[2]),
                (insert[0] + 1, insert[1], insert[2]),
            )
            scores[previous_index][current_index] = min(
                candidates,
                key=lambda candidate: (
                    candidate[0],
                    -candidate[1],
                    -candidate[2],
                ),
            )

    best: tuple[int, int, float, int] | None = None
    best_current_count = 0
    for current_index in range(1, current_count + 1):
        edits, matches, previous_start = scores[previous_count][current_index]
        span = max(previous_count - previous_start, current_index)
        similarity = 1 - edits / span
        if (
            matches < _MIN_OVERLAP_MATCH_TOKENS
            or similarity < _MIN_OVERLAP_SIMILARITY
        ):
            continue
        candidate = (matches - edits, matches, similarity, current_index)
        if best is None or candidate > best:
            best = candidate
            best_current_count = current_index

    if best is None:
        return current.strip()
    return current[current_tokens[best_current_count - 1][1] :].lstrip()


def _normalized_tokens(text: str) -> list[tuple[str, int]]:
    tokens = []
    for match in re.finditer(r"\S+", text):
        normalized = re.sub(r"[^\w]+", "", match.group(), flags=re.UNICODE).casefold()
        if normalized:
            tokens.append((normalized, match.end()))
    return tokens


def _format_timestamp(milliseconds: int) -> str:
    total_seconds = milliseconds // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _ffmpeg_time(milliseconds: int) -> str:
    return f"{milliseconds / 1000:.3f}"


def _normalize_transcript(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _validate_chunk(
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
    text = _normalize_transcript(chunk.text)
    if not text:
        raise ValueError(
            f"Cached audio transcription for {source!r} chunk {expected_index} "
            "is empty."
        )
    return chunk.model_copy(update={"text": text})


def _same_conversion(
    current: _AudioConversionManifest,
    expected: _AudioConversionManifest,
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


def _same_source_and_settings(
    manifest: _AudioConversionManifest,
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


def _load_manifest(path: Path) -> _AudioConversionManifest | None:
    if not path.exists():
        return None
    return _AudioConversionManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_model_atomic(path: Path, model: BaseModel) -> None:
    _write_text_atomic(path, model.model_dump_json(indent=2))


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)
