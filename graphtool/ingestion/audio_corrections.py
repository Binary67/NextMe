from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from graphtool.ingestion.audio_transcript import AudioTranscriptChunk
from graphtool.ingestion.cache_io import text_hash
from graphtool.llm.base import LLMClient
from graphtool.llm.types import LLMMessage

AUDIO_CORRECTION_REVISION = 1

_CORRECTION_SYSTEM_PROMPT = (
    "Find only proper-noun spelling discrepancies in an "
    "audio transcript. Propose exact substring replacements "
    "using the canonical terms supplied by the user. Do not "
    "change punctuation, grammar, numbers, or wording. Set "
    "decision to apply only when the intended canonical term "
    "is unambiguous; otherwise set it to review. Context must "
    "be the shortest exact excerpt from the transcript that "
    "contains one occurrence of original and uniquely "
    "identifies it."
)


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


def load_corrections(
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


def serialize_corrections(
    corrections: Sequence[AudioTranscriptCorrection],
) -> str:
    ordered = sorted(corrections, key=lambda item: (item.chunk, item.id))
    if not ordered:
        return ""
    return "\n".join(item.model_dump_json() for item in ordered) + "\n"


def correction_input_hash(
    chunks: Sequence[AudioTranscriptChunk],
    terms: Sequence[str],
    corrector: LLMClient,
) -> str:
    values = [
        corrector.text_model,
        str(AUDIO_CORRECTION_REVISION),
        *terms,
        *(f"{chunk.index}:{text_hash(chunk.text)}" for chunk in chunks),
    ]
    return text_hash("\0".join(values))


def generate_corrections(
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
                LLMMessage(role="system", content=_CORRECTION_SYSTEM_PROMPT),
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


def merge_corrections(
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


def apply_corrections(
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


def _correction_id(
    source: str,
    chunk: int,
    original: str,
    replacement: str,
    context: str,
) -> str:
    value = "\0".join((source, str(chunk), original, replacement, context))
    return text_hash(value)[:16]


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
