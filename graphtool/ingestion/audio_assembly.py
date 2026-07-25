import re
from collections.abc import Sequence

from graphtool.ingestion.audio_corrections import (
    AudioTranscriptCorrection,
    apply_corrections,
)
from graphtool.ingestion.audio_transcript import AudioTranscriptChunk

AUDIO_ASSEMBLY_REVISION = 1
_MIN_OVERLAP_MATCH_TOKENS = 5
_MAX_OVERLAP_WINDOW_TOKENS = 100
_MIN_OVERLAP_SIMILARITY = 0.65


def assemble_markdown(
    file_name: str,
    source: str,
    chunks: list[AudioTranscriptChunk],
    corrections: Sequence[AudioTranscriptCorrection],
    terms: Sequence[str],
) -> str:
    blocks = [f"# Transcript: {file_name}"]
    previous_text = ""
    for chunk in chunks:
        corrected_text = apply_corrections(
            source,
            chunk,
            corrections,
            terms,
        )
        text = remove_fuzzy_overlap(previous_text, corrected_text)
        if text:
            blocks.append(
                f"## {_format_timestamp(chunk.start_milliseconds)}\n\n{text}"
            )
        previous_text = corrected_text
    return "\n\n".join(blocks).rstrip() + "\n"


def remove_fuzzy_overlap(previous: str, current: str) -> str:
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
