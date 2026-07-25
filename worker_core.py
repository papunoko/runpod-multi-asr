"""Pure contracts for the fixed, three-backend Runpod ASR worker."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


JOB_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 2
SCHEMA_VERSION = JOB_SCHEMA_VERSION  # Backward-compatible name for job records.
BACKEND_ORDER = ("faster_whisper", "reazon_nemo", "qwen3")
MODEL_PATHS = {
    "faster_whisper": "/opt/models/whisper",
    "reazon_nemo": "/opt/models/reazon",
    "qwen3": "/opt/models/qwen",
}
BACKEND_MODEL_ARGUMENTS = {
    "faster_whisper": "/opt/models/whisper",
    "reazon_nemo": "/opt/models/reazon/reazonspeech-nemo-v2.nemo",
    "qwen3": "/opt/models/qwen",
}
MEETING_KEY_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,95}")
JOB_ID_RE = re.compile(r"[0-9a-f]{32}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
CHUNK_ID_RE = re.compile(r"chunk_[0-9]{3,6}")
JOB_STATUSES = frozenset({"accepted", "running", "complete", "failed"})
TERMINAL_STATUSES = frozenset({"complete", "failed"})
TRANSITIONS = {
    "accepted": frozenset({"running", "failed"}),
    "running": frozenset({"complete", "failed"}),
    "complete": frozenset(),
    "failed": frozenset(),
}
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_CHUNKS = 5000
MAX_TRANSCRIPT_BYTES = 16 * 1024 * 1024


class ContractError(ValueError):
    """Input or persisted state violated the worker contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def validate_meeting_key(value: str) -> str:
    if not isinstance(value, str) or not MEETING_KEY_RE.fullmatch(value):
        raise ContractError("meeting_key must match [a-z0-9][a-z0-9-]{0,95}")
    return value


def validate_job_id(value: str) -> str:
    if not isinstance(value, str) or not JOB_ID_RE.fullmatch(value):
        raise ContractError("job_id must be 32 lowercase hexadecimal characters")
    return value


def validate_sha256(value: str) -> str:
    normalized = value.lower() if isinstance(value, str) else ""
    if not SHA256_RE.fullmatch(normalized):
        raise ContractError("sha256 must be 64 hexadecimal characters")
    return normalized


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ContractError("JSON contains a duplicate key")
        value[key] = item
    return value


def _read_json(path: Path, label: str) -> Any:
    try:
        size = path.stat().st_size
        if size <= 0 or size > MAX_JSON_BYTES or not path.is_file() or path.is_symlink():
            raise ContractError(f"{label} has an invalid file contract")
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite value: {value}")
            ),
        )
    except ContractError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError(f"{label} is unreadable") from exc


def read_job_record(path: Path) -> dict[str, Any]:
    payload = _read_json(path, "job record")
    if not isinstance(payload, dict):
        raise ContractError("job record must be an object")
    if (
        type(payload.get("schemaVersion")) is not int
        or payload["schemaVersion"] != JOB_SCHEMA_VERSION
    ):
        raise ContractError("job record has an invalid schema version")
    validate_job_id(payload.get("jobId", ""))
    validate_meeting_key(payload.get("meetingKey", ""))
    validate_sha256(payload.get("audioSha256", ""))
    if payload.get("status") not in JOB_STATUSES:
        raise ContractError("job record has an invalid status")
    if type(payload.get("audioBytes")) is not int or payload["audioBytes"] <= 0:
        raise ContractError("job record has an invalid audio byte count")
    audio_duration = payload.get("audioDurationSeconds")
    if (
        not isinstance(audio_duration, (int, float))
        or isinstance(audio_duration, bool)
        or not math.isfinite(float(audio_duration))
        or float(audio_duration) <= 0
    ):
        raise ContractError("job record has an invalid audio duration")
    probed_duration = payload.get("probedAudioDurationSeconds")
    if probed_duration is not None and (
        not isinstance(probed_duration, (int, float))
        or isinstance(probed_duration, bool)
        or not math.isfinite(float(probed_duration))
        or float(probed_duration) <= 0
    ):
        raise ContractError("job record has an invalid probed audio duration")
    backends = payload.get("backends")
    if backends is not None:
        if not isinstance(backends, dict) or set(backends) != set(BACKEND_ORDER):
            raise ContractError("job record has an invalid backend set")
        for backend_id, state in backends.items():
            if (
                backend_id not in BACKEND_ORDER
                or not isinstance(state, dict)
                or state.get("status") not in {"pending", "running", "complete", "failed"}
            ):
                raise ContractError("job record has an invalid backend status")
    return payload


def new_job_record(
    *,
    job_id: str,
    meeting_key: str,
    audio_sha256: str,
    audio_bytes: int,
    audio_duration_seconds: float,
) -> dict[str, Any]:
    validate_job_id(job_id)
    validate_meeting_key(meeting_key)
    validate_sha256(audio_sha256)
    if not isinstance(audio_bytes, int) or isinstance(audio_bytes, bool) or audio_bytes <= 0:
        raise ContractError("audio_bytes must be a positive integer")
    if (
        not isinstance(audio_duration_seconds, (int, float))
        or isinstance(audio_duration_seconds, bool)
        or not math.isfinite(float(audio_duration_seconds))
        or float(audio_duration_seconds) <= 0
    ):
        raise ContractError("audio_duration_seconds must be finite and positive")
    now = utc_now()
    return {
        "schemaVersion": JOB_SCHEMA_VERSION,
        "jobId": job_id,
        "meetingKey": meeting_key,
        "audioSha256": audio_sha256,
        "audioBytes": audio_bytes,
        "audioDurationSeconds": float(audio_duration_seconds),
        "status": "accepted",
        "backendOrder": list(BACKEND_ORDER),
        "backends": {
            backend_id: {"status": "pending"} for backend_id in BACKEND_ORDER
        },
        "createdAt": now,
        "updatedAt": now,
        "errorCode": None,
    }


def transition_job(
    record: dict[str, Any], status: str, *, error_code: str | None = None
) -> dict[str, Any]:
    current = record.get("status")
    if current not in JOB_STATUSES or status not in JOB_STATUSES:
        raise ContractError("job transition contains an invalid status")
    if status not in TRANSITIONS[current]:
        raise ContractError(f"invalid job transition: {current} -> {status}")
    if status == "failed":
        if not isinstance(error_code, str) or not re.fullmatch(
            r"[a-z][a-z0-9_]{2,63}", error_code
        ):
            raise ContractError("failed status requires a safe error_code")
    elif error_code is not None:
        raise ContractError("error_code is only valid for failed status")
    updated = dict(record)
    updated["status"] = status
    updated["updatedAt"] = utc_now()
    updated["errorCode"] = error_code
    return updated


def load_model_identities(path: Path) -> tuple[dict[str, Any], ...]:
    payload = _read_json(path, "model identity manifest")
    if not isinstance(payload, dict) or set(payload) != {"schemaVersion", "backends"}:
        raise ContractError("model identity manifest has invalid fields")
    if type(payload["schemaVersion"]) is not int or payload["schemaVersion"] != 1:
        raise ContractError("model identity manifest has an invalid schema version")
    entries = payload["backends"]
    if not isinstance(entries, list) or len(entries) != len(BACKEND_ORDER):
        raise ContractError("model identity manifest must contain three backends")
    required = {
        "id",
        "label",
        "repository",
        "revision",
        "path",
        "manifestPath",
        "manifestSha256",
    }
    normalized: list[dict[str, Any]] = []
    for expected_id, entry in zip(BACKEND_ORDER, entries, strict=True):
        if not isinstance(entry, dict) or set(entry) != required:
            raise ContractError("model identity entry has invalid fields")
        if entry.get("id") != expected_id or entry.get("path") != MODEL_PATHS[expected_id]:
            raise ContractError("model identity order or path is invalid")
        expected_manifest = f"{MODEL_PATHS[expected_id]}/MODEL_MANIFEST.sha256"
        if entry.get("manifestPath") != expected_manifest:
            raise ContractError("model manifest path is invalid")
        for key in ("label", "repository", "revision"):
            value = entry.get(key)
            if not isinstance(value, str) or not value or len(value) > 512:
                raise ContractError("model identity text field is invalid")
        validate_sha256(entry.get("manifestSha256", ""))
        normalized.append(dict(entry))
    return tuple(normalized)


def _read_prepared_chunks(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "chunks.jsonl"
    try:
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size <= 0
            or path.stat().st_size > MAX_JSON_BYTES
        ):
            raise ContractError("prepared chunk manifest has an invalid file contract")
        rows = [
            json.loads(line, object_pairs_hook=_strict_object)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except ContractError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("prepared chunk manifest is unreadable") from exc
    if not rows or len(rows) > MAX_CHUNKS or not all(isinstance(row, dict) for row in rows):
        raise ContractError("prepared chunk manifest has an invalid row count")
    seen: set[str] = set()
    previous_start = -1.0
    required_fields = {
        "chunk_id",
        "index",
        "core_start_s",
        "core_end_s",
        "audio_start_s",
        "audio_end_s",
        "boundary_reason",
        "raw_audio",
        "enhanced_audio",
    }
    for index, row in enumerate(rows):
        chunk_id = row.get("chunk_id")
        start = row.get("audio_start_s")
        end = row.get("audio_end_s")
        if (
            set(row) != required_fields
            or not isinstance(chunk_id, str)
            or not CHUNK_ID_RE.fullmatch(chunk_id)
            or chunk_id in seen
            or type(row.get("index")) is not int
            or row["index"] != index
            or not isinstance(row.get("boundary_reason"), str)
            or not row["boundary_reason"]
            or row.get("raw_audio") != f"audio/raw_stereo/{chunk_id}.mp3"
            or row.get("enhanced_audio") != f"audio/enhanced_mono/{chunk_id}.wav"
            or isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
            or float(start) < previous_start
            or float(start) < 0
            or float(end) <= float(start)
        ):
            raise ContractError(f"prepared chunk row {index} is invalid")
        for field in ("core_start_s", "core_end_s"):
            value = row[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ContractError(f"prepared chunk row {index} is invalid")
        if float(row["core_end_s"]) <= float(row["core_start_s"]):
            raise ContractError(f"prepared chunk row {index} is invalid")
        for relative in (row["raw_audio"], row["enhanced_audio"]):
            candidate = run_dir.joinpath(*PurePosixPath(relative).parts)
            if not candidate.is_file() or candidate.is_symlink():
                raise ContractError("prepared chunk audio is missing or unsafe")
        seen.add(chunk_id)
        previous_start = float(start)
    return rows


def validate_prepared_run(run_dir: Path, record: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = _read_json(run_dir / "manifest.json", "prepared manifest")
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "source",
        "preprocess",
        "chunk_count",
    }:
        raise ContractError("prepared manifest has invalid fields")
    source = manifest.get("source")
    preprocess = manifest.get("preprocess")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
        or not isinstance(source, dict)
        or not isinstance(preprocess, dict)
        or source.get("sha256") != record.get("audioSha256")
        or source.get("path") != str(run_dir.parent / "audio.mp3")
    ):
        raise ContractError("prepared manifest does not match the job audio")
    try:
        duration = float(source["probe"]["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("prepared manifest has no valid source duration") from exc
    expected_duration = float(
        record.get("probedAudioDurationSeconds", record["audioDurationSeconds"])
    )
    if not math.isfinite(duration) or abs(duration - expected_duration) > max(
        5.0, expected_duration * 0.01
    ):
        raise ContractError("prepared duration does not match the job audio")
    fixed_settings = {
        "target_chunk_s": 75.0,
        "overlap_s": 3.0,
        "search_window_s": 15.0,
    }
    for key, expected in fixed_settings.items():
        value = preprocess.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) != expected
        ):
            raise ContractError("prepared manifest settings are not fixed")
    chunks = _read_prepared_chunks(run_dir)
    if type(manifest.get("chunk_count")) is not int or manifest["chunk_count"] != len(chunks):
        raise ContractError("prepared chunk count does not match the manifest")
    return chunks


def _bounded_text(value: Any, total: list[int]) -> str:
    if not isinstance(value, str):
        raise ContractError("backend text is invalid")
    encoded = len(value.encode("utf-8"))
    total[0] += encoded
    if encoded > 1024 * 1024 or total[0] > MAX_TRANSCRIPT_BYTES:
        raise ContractError("backend text exceeds the size limit")
    return value


def validate_backend_output(run_dir: Path, backend_id: str) -> dict[str, Any]:
    if backend_id not in BACKEND_ORDER:
        raise ContractError("backend id is not fixed")
    prepared = _read_prepared_chunks(run_dir)
    prepared_by_id = {str(item["chunk_id"]): item for item in prepared}
    expected_ids = tuple(prepared_by_id)
    directory = run_dir / "hypotheses" / backend_id
    if not directory.is_dir() or directory.is_symlink():
        raise ContractError("backend output directory is unavailable")
    expected_names = {"backend.json", *(f"{item}.json" for item in expected_ids)}
    entries = list(directory.iterdir())
    if any(not item.is_file() or item.is_symlink() for item in entries):
        raise ContractError("backend output directory contains an unsafe entry")
    if {item.name for item in entries} != expected_names:
        raise ContractError("backend output file set is incomplete or unexpected")

    summary = _read_json(directory / "backend.json", "backend summary")
    required_summary = {
        "schema_version",
        "backend_id",
        "model",
        "chunks_total",
        "chunks_completed",
        "failures",
        "context_snapshot_id",
        "audio_variant",
        "seed_backend",
    }
    if (
        not isinstance(summary, dict)
        or set(summary) != required_summary
        or type(summary.get("schema_version")) is not int
        or summary.get("schema_version") != 1
        or summary.get("backend_id") != backend_id
        or not isinstance(summary.get("model"), str)
        or summary["model"] != BACKEND_MODEL_ARGUMENTS[backend_id]
        or type(summary.get("chunks_total")) is not int
        or summary.get("chunks_total") != len(prepared)
        or type(summary.get("chunks_completed")) is not int
        or summary.get("chunks_completed") != len(prepared)
        or summary.get("failures") != []
    ):
        raise ContractError("backend summary did not prove a complete successful run")

    total_text = [0]
    chunk_results: list[dict[str, Any]] = []
    nonempty = 0
    segment_count = 0
    for chunk_id in expected_ids:
        raw = _read_json(directory / f"{chunk_id}.json", "backend chunk result")
        required_chunk = {
            "schema_version",
            "backend_id",
            "model",
            "chunk_id",
            "audio_start_s",
            "language",
            "capabilities",
            "text",
            "segments",
            "selected_term_ids",
            "context_snapshot_id",
            "metadata",
        }
        if (
            not isinstance(raw, dict)
            or set(raw) != required_chunk
            or type(raw.get("schema_version")) is not int
            or raw.get("schema_version") != 1
            or raw.get("backend_id") != backend_id
            or raw.get("chunk_id") != chunk_id
            or raw.get("model") != summary["model"]
            or not isinstance(raw.get("segments"), list)
            or not isinstance(raw.get("language"), str)
            or not isinstance(raw.get("capabilities"), list)
            or not isinstance(raw.get("selected_term_ids"), list)
            or not isinstance(raw.get("metadata"), dict)
        ):
            raise ContractError("backend chunk result has an invalid schema")
        text = _bounded_text(raw.get("text"), total_text)
        if text.strip():
            nonempty += 1
        prepared_row = prepared_by_id[chunk_id]
        audio_start = raw.get("audio_start_s")
        if (
            isinstance(audio_start, bool)
            or not isinstance(audio_start, (int, float))
            or not math.isfinite(float(audio_start))
            or abs(float(audio_start) - float(prepared_row["audio_start_s"])) > 0.001
        ):
            raise ContractError("backend chunk audio start does not match preparation")
        duration = float(prepared_row["audio_end_s"]) - float(
            prepared_row["audio_start_s"]
        )
        normalized_segments: list[dict[str, Any]] = []
        previous_start = -1.0
        for index, segment in enumerate(raw["segments"]):
            if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
                raise ContractError("backend segment has an invalid schema")
            start = segment.get("start")
            end = segment.get("end")
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or not math.isfinite(float(start))
                or not math.isfinite(float(end))
                or float(start) < previous_start
                or float(start) < 0
                or float(end) < float(start)
                or float(end) > duration + 1.0
            ):
                raise ContractError("backend segment timestamps are invalid")
            segment_text = _bounded_text(segment["text"], total_text)
            if backend_id == "faster_whisper" and not segment_text.strip():
                raise ContractError("Whisper timestamp segment text is empty")
            normalized_segments.append(
                {
                    "index": index,
                    "start": float(start),
                    "end": float(end),
                    "text": segment_text,
                }
            )
            previous_start = float(start)
            segment_count += 1
        if backend_id == "faster_whisper" and text.strip() and not normalized_segments:
            raise ContractError("Whisper text requires timestamp segments")
        chunk_results.append(
            {
                "chunkId": chunk_id,
                "audioStartSeconds": float(audio_start),
                "language": raw["language"],
                "text": text,
                "segments": normalized_segments,
            }
        )
    if nonempty == 0:
        raise ContractError("backend produced no transcript text")
    if backend_id == "faster_whisper" and segment_count == 0:
        raise ContractError("Whisper produced no timestamp segments")
    return {
        "summary": {
            "model": summary["model"],
            "chunksTotal": len(prepared),
            "chunksCompleted": len(prepared),
            "failures": [],
            "nonemptyChunks": nonempty,
            "segmentCount": segment_count,
        },
        "text": "\n".join(item["text"] for item in chunk_results if item["text"]),
        "chunks": chunk_results,
    }


def build_result(
    *,
    record: dict[str, Any],
    model_identities: Iterable[dict[str, Any]],
    backend_outputs: dict[str, dict[str, Any]],
    backend_timings: dict[str, dict[str, Any]],
    total_processing_seconds: float,
) -> dict[str, Any]:
    if record.get("status") != "running":
        raise ContractError("result can only be built for a running job")
    if (
        isinstance(total_processing_seconds, bool)
        or not isinstance(total_processing_seconds, (int, float))
        or not math.isfinite(float(total_processing_seconds))
        or float(total_processing_seconds) <= 0
    ):
        raise ContractError("total processing seconds is invalid")
    identities = tuple(model_identities)
    if tuple(item.get("id") for item in identities) != BACKEND_ORDER:
        raise ContractError("result model identities are incomplete or unordered")
    if tuple(backend_outputs) != BACKEND_ORDER or tuple(backend_timings) != BACKEND_ORDER:
        raise ContractError("result backends are incomplete or unordered")
    backends: list[dict[str, Any]] = []
    for identity in identities:
        backend_id = identity["id"]
        output = backend_outputs[backend_id]
        timing = backend_timings[backend_id]
        if set(timing) != {"startedAt", "completedAt", "processingSeconds"}:
            raise ContractError("backend timing fields are invalid")
        elapsed = timing["processingSeconds"]
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) <= 0
            or not isinstance(timing["startedAt"], str)
            or not isinstance(timing["completedAt"], str)
        ):
            raise ContractError("backend timing is invalid")
        if set(output) != {"summary", "text", "chunks"}:
            raise ContractError("validated backend output is malformed")
        model = {
            key: identity[key]
            for key in (
                "label",
                "repository",
                "revision",
                "path",
                "manifestPath",
                "manifestSha256",
            )
        }
        backends.append(
            {
                "id": backend_id,
                "status": "complete",
                "model": model,
                **timing,
                **output,
            }
        )
    return {
        "schemaVersion": RESULT_SCHEMA_VERSION,
        "jobId": record["jobId"],
        "meetingKey": record["meetingKey"],
        "status": "complete",
        "audioSha256": record["audioSha256"],
        "audioBytes": record["audioBytes"],
        "audioDurationSeconds": float(
            record.get("probedAudioDurationSeconds", record["audioDurationSeconds"])
        ),
        "backendOrder": list(BACKEND_ORDER),
        "backends": backends,
        "totalProcessingSeconds": float(total_processing_seconds),
        "completedAt": utc_now(),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model_manifest(model_root: Path, manifest_path: Path, expected_sha256: str) -> None:
    validate_sha256(expected_sha256)
    if (
        not model_root.is_dir()
        or model_root.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
        or manifest_path.parent != model_root
        or manifest_path.name != "MODEL_MANIFEST.sha256"
        or manifest_path.stat().st_size <= 0
        or manifest_path.stat().st_size > 1024 * 1024
        or sha256_file(manifest_path) != expected_sha256
    ):
        raise ContractError("model manifest identity check failed")
    try:
        root = model_root.resolve(strict=True)
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ContractError("model manifest is unreadable") from exc
    if not lines or len(lines) > 10000:
        raise ContractError("model manifest entry count is invalid")
    seen: set[str] = set()
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64}) [ *]([^\x00-\x1f]+)", line)
        if not match:
            raise ContractError("model manifest line is invalid")
        relative_text = match.group(2)
        relative = PurePosixPath(relative_text)
        if (
            relative.is_absolute()
            or relative_text in seen
            or any(part in {"", ".", ".."} for part in relative.parts)
            or "\\" in relative_text
        ):
            raise ContractError("model manifest path is unsafe")
        seen.add(relative_text)
        candidate = model_root.joinpath(*relative.parts)
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as exc:
            raise ContractError("model manifest entry escaped its root") from exc
        if not candidate.is_file() or candidate.is_symlink():
            raise ContractError("model manifest entry is not a regular file")
        if sha256_file(candidate) != match.group(1):
            raise ContractError("model file hash mismatch")
    actual_files: set[str] = set()
    for candidate in model_root.rglob("*"):
        if _is_link_like_for_manifest(candidate):
            raise ContractError("model directory contains a link")
        if candidate.is_file():
            actual_files.add(candidate.relative_to(model_root).as_posix())
        elif not candidate.is_dir():
            raise ContractError("model directory contains an unsafe entry")
    if actual_files != seen | {"MODEL_MANIFEST.sha256"}:
        raise ContractError("model directory does not exactly match its manifest")


def _is_link_like_for_manifest(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())
