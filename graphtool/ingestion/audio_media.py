import math
import subprocess
from pathlib import Path

AUDIO_CHUNK_MILLISECONDS = 20 * 60 * 1000
AUDIO_OVERLAP_MILLISECONDS = 5 * 1000
AUDIO_SAMPLE_RATE = 16_000
AUDIO_BITRATE = "64k"
AUDIO_MAX_CHUNK_BYTES = 24_000_000


def probe_duration(audio_path: Path, source: str, ffprobe: str) -> int:
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


def chunk_boundaries(duration_milliseconds: int) -> list[tuple[int, int]]:
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


def render_chunk(
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


def _ffmpeg_time(milliseconds: int) -> str:
    return f"{milliseconds / 1000:.3f}"
