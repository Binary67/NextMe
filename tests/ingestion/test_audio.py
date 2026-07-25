import json

import pytest

from graphtool.ingestion import (
    audio,
    audio_assembly,
    audio_cache,
    audio_media,
)
from graphtool.ingestion.audio import convert_audio_to_markdown
from graphtool.ingestion.audio_glossary import (
    load_audio_transcription_terms,
)
from graphtool.source import source_key


class FakeTranscriber:
    def __init__(self, responses, *, model="transcription-deployment"):
        self.transcription_model = model
        self.responses = list(responses)
        self.calls = []

    def transcribe_audio(self, path, *, prompt=None):
        self.calls.append((path.read_bytes(), prompt))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeCorrector:
    def __init__(self, responses=None, *, model="fast-deployment"):
        self.text_model = model
        self.responses = list(responses or [])
        self.calls = []

    def generate_structured(self, messages, response_model):
        self.calls.append((messages, response_model))
        if not self.responses:
            return response_model(corrections=[])
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response_model.model_validate(response)


def _hip_sa_proposals(*, decision="apply", replacement="HIP-SA"):
    return {
        "corrections": [
            {
                "original": "hip saw",
                "replacement": replacement,
                "context": "We reviewed the hip saw roadmap.",
                "decision": decision,
            }
        ]
    }


def _set_correction_decision(path, decision, *, reviewed):
    correction = json.loads(path.read_text())
    correction["decision"] = decision
    correction["reviewed"] = reviewed
    path.write_text(json.dumps(correction) + "\n")


def _prepare(monkeypatch, duration_milliseconds):
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        audio,
        "probe_duration",
        lambda path, source, ffprobe: duration_milliseconds,
    )
    render_calls = []

    def fake_render(
        source_path,
        chunk_path,
        start_milliseconds,
        end_milliseconds,
        ffmpeg,
        source,
    ):
        render_calls.append((start_milliseconds, end_milliseconds))
        chunk_path.write_bytes(
            f"audio-{start_milliseconds}-{end_milliseconds}".encode()
        )

    monkeypatch.setattr(audio, "render_chunk", fake_render)
    return render_calls


def test_convert_audio_chunks_transcribes_and_assembles_markdown(
    monkeypatch,
    tmp_path,
):
    render_calls = _prepare(monkeypatch, 40 * 60 * 1000)
    path = tmp_path / "quarterly-review.mp3"
    path.write_bytes(b"original-audio")
    first_transcript = "Opening facts. Shared boundary words remain exactly the same."
    second_transcript = (
        "Shared boundary words remain exactly the same. New section facts."
    )
    transcriber = FakeTranscriber([first_transcript, second_transcript])
    terms = ["HIP-SA", "Aishwarya Rao"]

    markdown = convert_audio_to_markdown(
        path,
        "documents/recordings/quarterly-review.mp3",
        transcriber,
        FakeCorrector(),
        tmp_path / "cache",
        terms,
    )

    assert render_calls == [
        (0, 1_205_000),
        (1_200_000, 2_400_000),
    ]
    glossary_prompt = (
        "Expected proper nouns and exact spellings:\n"
        "- HIP-SA\n"
        "- Aishwarya Rao\n"
        "Use these spellings only when they match the spoken audio."
    )
    assert transcriber.calls[0][1] == glossary_prompt
    assert transcriber.calls[1][1] == (
        f"{glossary_prompt}\n\nPrevious transcript context:\n{first_transcript}"
    )
    assert markdown == (
        "# Transcript: quarterly-review.mp3\n\n"
        "## 00:00:00\n\n"
        "Opening facts. Shared boundary words remain exactly the same.\n\n"
        "## 00:20:00\n\n"
        "New section facts.\n"
    )


def test_convert_audio_uses_complete_cache_without_external_tools(
    monkeypatch,
    tmp_path,
):
    _prepare(monkeypatch, 60_000)
    path = tmp_path / "recording.mp3"
    path.write_bytes(b"original-audio")
    cache_dir = tmp_path / "cache"
    expected = convert_audio_to_markdown(
        path,
        "documents/recordings/recording.mp3",
        FakeTranscriber(["Cached transcript."]),
        FakeCorrector(),
        cache_dir,
        [],
    )
    monkeypatch.setattr(
        audio.shutil,
        "which",
        lambda name: pytest.fail("completed cache looked up external tools"),
    )
    transcriber = FakeTranscriber([])

    actual = convert_audio_to_markdown(
        path,
        "documents/recordings/recording.mp3",
        transcriber,
        FakeCorrector(),
        cache_dir,
        [],
    )

    assert actual == expected
    assert transcriber.calls == []


def test_convert_audio_resumes_completed_chunks_after_failure(
    monkeypatch,
    tmp_path,
):
    render_calls = _prepare(monkeypatch, 40 * 60 * 1000)
    path = tmp_path / "recording.mp3"
    path.write_bytes(b"original-audio")
    cache_dir = tmp_path / "cache"
    source = "documents/recordings/recording.mp3"

    with pytest.raises(RuntimeError, match="request failed"):
        convert_audio_to_markdown(
            path,
            source,
            FakeTranscriber([
                "First chunk.",
                RuntimeError("request failed"),
            ]),
            FakeCorrector(),
            cache_dir,
            [],
        )

    resumed = FakeTranscriber(["Second chunk."])
    markdown = convert_audio_to_markdown(
        path,
        source,
        resumed,
        FakeCorrector(),
        cache_dir,
        [],
    )

    assert render_calls == [
        (0, 1_205_000),
        (1_200_000, 2_400_000),
        (1_200_000, 2_400_000),
    ]
    assert len(resumed.calls) == 1
    assert resumed.calls[0][1].endswith("First chunk.")
    assert "First chunk." in markdown
    assert "Second chunk." in markdown


def test_convert_audio_invalidates_cache_when_model_changes(monkeypatch, tmp_path):
    render_calls = _prepare(monkeypatch, 60_000)
    path = tmp_path / "recording.mp3"
    path.write_bytes(b"original-audio")
    cache_dir = tmp_path / "cache"
    source = "documents/recordings/recording.mp3"
    convert_audio_to_markdown(
        path,
        source,
        FakeTranscriber(["First."], model="transcribe-a"),
        FakeCorrector(),
        cache_dir,
        [],
    )

    markdown = convert_audio_to_markdown(
        path,
        source,
        FakeTranscriber(["Second."], model="transcribe-b"),
        FakeCorrector(),
        cache_dir,
        [],
    )

    assert render_calls == [(0, 60_000), (0, 60_000)]
    assert "Second." in markdown
    manifest_path = cache_dir / source_key(source) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["model"] == "transcribe-b"
    assert manifest["assembly_revision"] == audio_assembly.AUDIO_ASSEMBLY_REVISION
    assert manifest["complete"] is True
    assert manifest["markdown_hash"]


def test_convert_audio_reassembles_from_raw_chunks_when_assembly_changes(
    monkeypatch,
    tmp_path,
):
    render_calls = _prepare(monkeypatch, 40 * 60 * 1000)
    path = tmp_path / "recording.mp3"
    path.write_bytes(b"original-audio")
    cache_dir = tmp_path / "cache"
    source = "documents/recordings/recording.mp3"
    convert_audio_to_markdown(
        path,
        source,
        FakeTranscriber([
            "Opening facts. Shared boundary words remain exactly the same.",
            "Shared boundary words remain exactly the same. New facts.",
        ]),
        FakeCorrector(),
        cache_dir,
        [],
    )
    monkeypatch.setattr(
        audio_cache,
        "AUDIO_ASSEMBLY_REVISION",
        audio_assembly.AUDIO_ASSEMBLY_REVISION + 1,
    )
    transcriber = FakeTranscriber([])

    markdown = convert_audio_to_markdown(
        path,
        source,
        transcriber,
        FakeCorrector(),
        cache_dir,
        [],
    )

    assert render_calls == [(0, 1_205_000), (1_200_000, 2_400_000)]
    assert transcriber.calls == []
    assert markdown.count("Shared boundary words remain exactly the same.") == 1


def test_convert_audio_fails_clearly_without_ffprobe(monkeypatch, tmp_path):
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    path = tmp_path / "recording.mp3"
    path.write_bytes(b"audio")

    with pytest.raises(RuntimeError, match="ffprobe was not found"):
        convert_audio_to_markdown(
            path,
            "documents/recordings/recording.mp3",
            FakeTranscriber([]),
            FakeCorrector(),
            tmp_path / "cache",
            [],
        )


def test_load_audio_transcription_terms_strips_canonical_terms(tmp_path):
    glossary_path = tmp_path / "transcription_glossary.json"
    glossary_path.write_text(
        json.dumps(
            {
                "terms": [
                    " HIP-SA ",
                    "Aishwarya Rao",
                ]
            }
        )
    )

    assert load_audio_transcription_terms(glossary_path) == [
        "HIP-SA",
        "Aishwarya Rao",
    ]


def test_load_audio_transcription_terms_returns_empty_for_missing_file(
    tmp_path,
):
    assert load_audio_transcription_terms(
        tmp_path / "transcription_glossary.json"
    ) == []


def test_convert_audio_invalidates_cache_when_glossary_changes(
    monkeypatch,
    tmp_path,
):
    render_calls = _prepare(monkeypatch, 60_000)
    path = tmp_path / "recording.mp3"
    path.write_bytes(b"original-audio")
    cache_dir = tmp_path / "cache"
    source = "documents/recordings/recording.mp3"
    convert_audio_to_markdown(
        path,
        source,
        FakeTranscriber(["First."]),
        FakeCorrector(),
        cache_dir,
        ["HIP-SA"],
    )
    transcriber = FakeTranscriber(["Second."])

    markdown = convert_audio_to_markdown(
        path,
        source,
        transcriber,
        FakeCorrector(),
        cache_dir,
        ["HIP-ZA"],
    )

    assert render_calls == [(0, 60_000), (0, 60_000)]
    assert "Second." in markdown
    assert "- HIP-ZA" in transcriber.calls[0][1]
    manifest_path = cache_dir / source_key(source) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["glossary_hash"]


def test_convert_audio_applies_validated_correction_and_preserves_raw_chunk(
    monkeypatch,
    tmp_path,
):
    _prepare(monkeypatch, 60_000)
    path = tmp_path / "recording.mp3"
    path.write_bytes(b"original-audio")
    cache_dir = tmp_path / "cache"
    source = "documents/recordings/recording.mp3"
    corrector = FakeCorrector([_hip_sa_proposals()])

    markdown = convert_audio_to_markdown(
        path,
        source,
        FakeTranscriber(["We reviewed the hip saw roadmap."]),
        corrector,
        cache_dir,
        ["HIP-SA"],
    )

    source_cache_dir = cache_dir / source_key(source)
    raw_chunk = json.loads(
        (source_cache_dir / "chunks" / "00000.json").read_text()
    )
    corrections = [
        json.loads(line)
        for line in (source_cache_dir / "corrections.jsonl")
        .read_text()
        .splitlines()
    ]
    manifest = json.loads((source_cache_dir / "manifest.json").read_text())
    assert "We reviewed the HIP-SA roadmap." in markdown
    assert raw_chunk["text"] == "We reviewed the hip saw roadmap."
    assert corrections[0]["decision"] == "apply"
    assert corrections[0]["reviewed"] is False
    assert corrections[0]["replacement"] == "HIP-SA"
    assert manifest["correction_model"] == "fast-deployment"
    assert manifest["correction_input_hash"]
    assert manifest["corrections_hash"]
    assert len(corrector.calls) == 1
    assert "Canonical terms:\n- HIP-SA" in corrector.calls[0][0][1].content


def test_convert_audio_records_review_correction_without_applying_it(
    monkeypatch,
    tmp_path,
):
    _prepare(monkeypatch, 60_000)
    path = tmp_path / "recording.mp3"
    path.write_bytes(b"original-audio")
    cache_dir = tmp_path / "cache"
    source = "documents/recordings/recording.mp3"
    corrector = FakeCorrector([
        {
            "corrections": [
                {
                    "original": "John",
                    "replacement": "Jon",
                    "context": "John presented the update.",
                    "decision": "review",
                }
            ]
        }
    ])

    markdown = convert_audio_to_markdown(
        path,
        source,
        FakeTranscriber(["John presented the update."]),
        corrector,
        cache_dir,
        ["Jon"],
    )

    correction_path = cache_dir / source_key(source) / "corrections.jsonl"
    correction = json.loads(correction_path.read_text())
    assert "John presented the update." in markdown
    assert correction["decision"] == "review"


def test_editing_correction_ledger_rebuilds_without_retranscription(
    monkeypatch,
    tmp_path,
):
    _prepare(monkeypatch, 60_000)
    path = tmp_path / "recording.mp3"
    path.write_bytes(b"original-audio")
    cache_dir = tmp_path / "cache"
    source = "documents/recordings/recording.mp3"
    source_cache_dir = cache_dir / source_key(source)
    convert_audio_to_markdown(
        path,
        source,
        FakeTranscriber(["We reviewed the hip saw roadmap."]),
        FakeCorrector([_hip_sa_proposals()]),
        cache_dir,
        ["HIP-SA"],
    )
    correction_path = source_cache_dir / "corrections.jsonl"
    _set_correction_decision(correction_path, "reject", reviewed=True)
    monkeypatch.setattr(
        audio.shutil,
        "which",
        lambda name: pytest.fail("ledger edit looked up external tools"),
    )
    transcriber = FakeTranscriber([])
    corrector = FakeCorrector()

    markdown = convert_audio_to_markdown(
        path,
        source,
        transcriber,
        corrector,
        cache_dir,
        ["HIP-SA"],
    )

    assert "We reviewed the hip saw roadmap." in markdown
    assert "HIP-SA" not in markdown
    assert transcriber.calls == []
    assert corrector.calls == []


def test_deleted_correction_row_reverts_derived_transcript(
    monkeypatch,
    tmp_path,
):
    _prepare(monkeypatch, 60_000)
    path = tmp_path / "recording.mp3"
    path.write_bytes(b"original-audio")
    cache_dir = tmp_path / "cache"
    source = "documents/recordings/recording.mp3"
    source_cache_dir = cache_dir / source_key(source)
    convert_audio_to_markdown(
        path,
        source,
        FakeTranscriber(["We reviewed the hip saw roadmap."]),
        FakeCorrector([_hip_sa_proposals()]),
        cache_dir,
        ["HIP-SA"],
    )
    (source_cache_dir / "corrections.jsonl").write_text("")
    monkeypatch.setattr(
        audio.shutil,
        "which",
        lambda name: pytest.fail("ledger edit looked up external tools"),
    )

    markdown = convert_audio_to_markdown(
        path,
        source,
        FakeTranscriber([]),
        FakeCorrector(),
        cache_dir,
        ["HIP-SA"],
    )

    assert "We reviewed the hip saw roadmap." in markdown
    assert "HIP-SA" not in markdown


def test_reject_decision_survives_correction_model_regeneration(
    monkeypatch,
    tmp_path,
):
    render_calls = _prepare(monkeypatch, 60_000)
    path = tmp_path / "recording.mp3"
    path.write_bytes(b"original-audio")
    cache_dir = tmp_path / "cache"
    source = "documents/recordings/recording.mp3"
    source_cache_dir = cache_dir / source_key(source)
    proposal = _hip_sa_proposals()
    convert_audio_to_markdown(
        path,
        source,
        FakeTranscriber(["We reviewed the hip saw roadmap."]),
        FakeCorrector([proposal], model="fast-a"),
        cache_dir,
        ["HIP-SA"],
    )
    correction_path = source_cache_dir / "corrections.jsonl"
    _set_correction_decision(correction_path, "reject", reviewed=True)
    transcriber = FakeTranscriber([])
    corrector = FakeCorrector([proposal], model="fast-b")

    markdown = convert_audio_to_markdown(
        path,
        source,
        transcriber,
        corrector,
        cache_dir,
        ["HIP-SA"],
    )

    retained = json.loads(correction_path.read_text())
    assert "We reviewed the hip saw roadmap." in markdown
    assert retained["decision"] == "reject"
    assert retained["reviewed"] is True
    assert transcriber.calls == []
    assert len(corrector.calls) == 1
    assert render_calls == [(0, 60_000)]


def test_correction_model_cannot_insert_non_glossary_replacement(
    monkeypatch,
    tmp_path,
):
    _prepare(monkeypatch, 60_000)
    path = tmp_path / "recording.mp3"
    path.write_bytes(b"original-audio")
    cache_dir = tmp_path / "cache"
    source = "documents/recordings/recording.mp3"

    markdown = convert_audio_to_markdown(
        path,
        source,
        FakeTranscriber(["We reviewed the hip saw roadmap."]),
        FakeCorrector([_hip_sa_proposals(replacement="HYP-SA")]),
        cache_dir,
        ["HIP-SA"],
    )

    correction_path = cache_dir / source_key(source) / "corrections.jsonl"
    assert "We reviewed the hip saw roadmap." in markdown
    assert correction_path.read_text() == ""


def test_render_chunk_normalizes_audio_and_checks_size(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        audio_media.Path(command[-1]).write_bytes(b"normalized-audio")

    monkeypatch.setattr(audio_media.subprocess, "run", fake_run)
    output_path = tmp_path / "chunk.mp3"

    audio_media.render_chunk(
        tmp_path / "source.wav",
        output_path,
        1_000,
        6_000,
        "/usr/bin/ffmpeg",
        "documents/recordings/source.wav",
    )

    command, kwargs = calls[0]
    assert command[:7] == [
        "/usr/bin/ffmpeg",
        "-v",
        "error",
        "-y",
        "-ss",
        "1.000",
        "-i",
    ]
    assert command[command.index("-t") + 1] == "5.000"
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-b:a") + 1] == "64k"
    assert kwargs == {"check": True, "capture_output": True, "text": True}


def test_chunk_boundaries_do_not_create_tiny_overlap_only_chunk():
    assert audio_media.chunk_boundaries(1_203_000) == [(0, 1_203_000)]
    assert audio_media.chunk_boundaries(2_403_000) == [
        (0, 1_205_000),
        (1_200_000, 2_403_000),
    ]


def test_fuzzy_overlap_removes_exact_boundary_text():
    previous = "Opening facts. Shared boundary words remain exactly the same."
    current = "Shared boundary words remain exactly the same. New section facts."

    result = audio_assembly.remove_fuzzy_overlap(previous, current)

    assert result == "New section facts."


def test_fuzzy_overlap_handles_asr_substitutions_and_insertions():
    previous = (
        "The report confirms revenue grew by twenty percent compared with last year."
    )
    current = (
        "Revenue grew by 20% compared with the last year. Margins also improved."
    )

    result = audio_assembly.remove_fuzzy_overlap(previous, current)

    assert result == "Margins also improved."


def test_fuzzy_overlap_preserves_current_text_when_confidence_is_low():
    previous = "Nothing related appears at this particular recording boundary."
    current = "A completely different section begins with unique material."

    result = audio_assembly.remove_fuzzy_overlap(previous, current)

    assert result == current
