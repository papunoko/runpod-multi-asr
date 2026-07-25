from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .utils import sha256_file, write_json_atomic, write_jsonl_atomic


SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
SILENCE_END_RE = re.compile(
    r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)"
)


# Stream-copying audio into a container for a different codec can fail (for
# example MP3 frames cannot be copied into an M4A container).  Keep the source
# container for the unfiltered transport copy; the enhanced path is always WAV.
COPY_SAFE_AUDIO_SUFFIXES = {
    ".aac",
    ".flac",
    ".m4a",
    ".mka",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}


@dataclass(frozen=True)
class Silence:
    start_s: float
    end_s: float
    duration_s: float

    @property
    def midpoint_s(self) -> float:
        return (self.start_s + self.end_s) / 2


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    index: int
    core_start_s: float
    core_end_s: float
    audio_start_s: float
    audio_end_s: float
    boundary_reason: str
    raw_audio: str
    enhanced_audio: str


def _require_binary(name: str) -> str:
    binary = shutil.which(name)
    if not binary:
        raise RuntimeError(f"Required executable not found: {name}")
    return binary


def _run(command: list[str], *, capture_stderr: bool = False) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stderr if capture_stderr else completed.stdout


def probe_audio(path: Path) -> dict[str, Any]:
    ffprobe = _require_binary("ffprobe")
    raw = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            (
                "format=filename,duration,size,bit_rate,format_name,tags:"
                "stream=index,codec_name,codec_type,sample_rate,channels,"
                "channel_layout,bit_rate"
            ),
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(raw)


def duration_seconds(probe: dict[str, Any]) -> float:
    return float(probe["format"]["duration"])


def detect_silences(
    path: Path, *, threshold_db: float = -35.0, min_silence_s: float = 0.8
) -> list[Silence]:
    ffmpeg = _require_binary("ffmpeg")
    output = _run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            f"silencedetect=noise={threshold_db}dB:d={min_silence_s}",
            "-f",
            "null",
            "-",
        ],
        capture_stderr=True,
    )
    silences: list[Silence] = []
    pending_start: float | None = None
    for line in output.splitlines():
        if match := SILENCE_START_RE.search(line):
            pending_start = float(match.group(1))
        if match := SILENCE_END_RE.search(line):
            end = float(match.group(1))
            measured_duration = float(match.group(2))
            start = pending_start if pending_start is not None else end - measured_duration
            if end > start:
                silences.append(Silence(start, end, end - start))
            pending_start = None
    return silences


def choose_core_boundaries(
    *,
    duration_s: float,
    silences: list[Silence],
    target_chunk_s: float = 360.0,
    search_window_s: float = 45.0,
) -> tuple[list[float], list[str]]:
    if target_chunk_s <= 30:
        raise ValueError("target_chunk_s must be greater than 30 seconds")
    boundaries = [0.0]
    reasons = ["start"]
    ideal = target_chunk_s
    while duration_s - ideal > target_chunk_s * 0.45:
        candidates = [
            silence
            for silence in silences
            if abs(silence.midpoint_s - ideal) <= search_window_s
            and silence.midpoint_s - boundaries[-1] >= target_chunk_s * 0.45
        ]
        if candidates:
            selected = min(
                candidates,
                key=lambda item: (
                    abs(item.midpoint_s - ideal),
                    -item.duration_s,
                ),
            )
            boundary = selected.midpoint_s
            reason = (
                f"silence_near_target:{selected.start_s:.3f}-"
                f"{selected.end_s:.3f}"
            )
        else:
            boundary = ideal
            reason = "fixed_target_no_silence"
        if duration_s - boundary < target_chunk_s * 0.45:
            break
        boundaries.append(round(boundary, 6))
        reasons.append(reason)
        ideal = boundary + target_chunk_s
    boundaries.append(duration_s)
    reasons.append("end")
    return boundaries, reasons


def _extract_raw_chunk(
    source: Path, destination: Path, *, start_s: float, end_s: float
) -> None:
    ffmpeg = _require_binary("ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_s:.6f}",
            "-t",
            f"{end_s - start_s:.6f}",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            "copy",
            "-map_metadata",
            "-1",
            "-avoid_negative_ts",
            "make_zero",
            str(destination),
        ]
    )


def _extract_enhanced_chunk(
    source: Path,
    destination: Path,
    *,
    start_s: float,
    end_s: float,
    normalize: bool,
) -> None:
    ffmpeg = _require_binary("ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    filters = ["highpass=f=70"]
    if normalize:
        filters.append("loudnorm=I=-20:LRA=9:TP=-2")
    _run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_s:.6f}",
            "-t",
            f"{end_s - start_s:.6f}",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-af",
            ",".join(filters),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-map_metadata",
            "-1",
            str(destination),
        ]
    )


def prepare_audio(
    source: Path,
    run_dir: Path,
    *,
    target_chunk_s: float = 360.0,
    overlap_s: float = 8.0,
    silence_threshold_db: float = -35.0,
    min_silence_s: float = 0.8,
    search_window_s: float = 45.0,
    normalize: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    run_dir.mkdir(parents=True, exist_ok=True)
    probe = probe_audio(source)
    duration_s = duration_seconds(probe)
    silences = detect_silences(
        source,
        threshold_db=silence_threshold_db,
        min_silence_s=min_silence_s,
    )
    boundaries, reasons = choose_core_boundaries(
        duration_s=duration_s,
        silences=silences,
        target_chunk_s=target_chunk_s,
        search_window_s=search_window_s,
    )

    chunks: list[Chunk] = []
    raw_dir = run_dir / "audio" / "raw_stereo"
    enhanced_dir = run_dir / "audio" / "enhanced_mono"
    raw_suffix = source.suffix.lower()
    if raw_suffix not in COPY_SAFE_AUDIO_SUFFIXES:
        # Matroska accepts the common audio codecs while preserving the source
        # bitstream.  This branch also avoids guessing a misleading extension.
        raw_suffix = ".mka"
    for index, (core_start, core_end) in enumerate(zip(boundaries, boundaries[1:])):
        audio_start = max(0.0, core_start - (overlap_s if index else 0.0))
        audio_end = min(
            duration_s,
            core_end + (overlap_s if index < len(boundaries) - 2 else 0.0),
        )
        chunk_id = f"chunk_{index:03d}"
        raw_path = raw_dir / f"{chunk_id}{raw_suffix}"
        enhanced_path = enhanced_dir / f"{chunk_id}.wav"
        if force or not raw_path.exists():
            _extract_raw_chunk(source, raw_path, start_s=audio_start, end_s=audio_end)
        if force or not enhanced_path.exists():
            _extract_enhanced_chunk(
                source,
                enhanced_path,
                start_s=audio_start,
                end_s=audio_end,
                normalize=normalize,
            )
        if raw_path.stat().st_size >= 25_000_000:
            raise RuntimeError(f"Raw chunk exceeds OpenAI's 25 MB limit: {raw_path}")
        if enhanced_path.stat().st_size >= 25_000_000:
            raise RuntimeError(f"Enhanced chunk exceeds OpenAI's 25 MB limit: {enhanced_path}")
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                index=index,
                core_start_s=core_start,
                core_end_s=core_end,
                audio_start_s=audio_start,
                audio_end_s=audio_end,
                boundary_reason=reasons[index + 1],
                raw_audio=str(raw_path.relative_to(run_dir)),
                enhanced_audio=str(enhanced_path.relative_to(run_dir)),
            )
        )

    manifest = {
        "schema_version": 1,
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "probe": probe,
        },
        "preprocess": {
            "target_chunk_s": target_chunk_s,
            "overlap_s": overlap_s,
            "silence_threshold_db": silence_threshold_db,
            "min_silence_s": min_silence_s,
            "search_window_s": search_window_s,
            "enhanced_filter": (
                "highpass=f=70,loudnorm=I=-20:LRA=9:TP=-2"
                if normalize
                else "highpass=f=70"
            ),
            "silence_count": len(silences),
            "silence_total_s": round(sum(item.duration_s for item in silences), 3),
        },
        "chunk_count": len(chunks),
    }
    write_json_atomic(run_dir / "manifest.json", manifest)
    write_jsonl_atomic(run_dir / "chunks.jsonl", (asdict(chunk) for chunk in chunks))
    write_json_atomic(
        run_dir / "silences.json",
        [asdict(item) | {"midpoint_s": item.midpoint_s} for item in silences],
    )
    return manifest | {"chunks": [asdict(chunk) for chunk in chunks]}

