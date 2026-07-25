from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

AUDIO_CONTEXT_TAIL_CHARS = 500


class _AudioTranscriptionGlossary(BaseModel):
    terms: list[str]


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


def glossary_prompt(terms: Sequence[str]) -> str | None:
    if not terms:
        return None
    term_list = "\n".join(f"- {term}" for term in terms)
    return (
        "Expected proper nouns and exact spellings:\n"
        f"{term_list}\n"
        "Use these spellings only when they match the spoken audio."
    )


def context_prompt(
    glossary_prompt_text: str | None,
    previous_text: str | None,
) -> str | None:
    previous_context = (
        previous_text[-AUDIO_CONTEXT_TAIL_CHARS:]
        if previous_text is not None
        else None
    )
    if glossary_prompt_text is None:
        return previous_context
    if previous_context is None:
        return glossary_prompt_text
    return (
        f"{glossary_prompt_text}\n\nPrevious transcript context:\n{previous_context}"
    )
