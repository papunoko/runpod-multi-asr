from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable

from .context import (
    build_transcription_prompt,
    context_snapshot_id,
    load_context,
    select_terms,
)
from .utils import read_json, read_jsonl, write_json_atomic


SCHEMA_VERSION = 1


def result_path(run_dir: Path, backend_id: str, chunk_id: str) -> Path:
    return run_dir / "hypotheses" / backend_id / f"{chunk_id}.json"


def load_backend_result(
    run_dir: Path, backend_id: str, chunk_id: str
) -> dict[str, Any] | None:
    path = result_path(run_dir, backend_id, chunk_id)
    return read_json(path) if path.exists() else None


def normalize_result(
    *,
    backend_id: str,
    model: str,
    chunk: dict[str, Any],
    result: dict[str, Any],
    selected_term_ids: list[str],
    context_id: str,
) -> dict[str, Any]:
    text = str(result.get("text", "")).strip()
    segments: list[dict[str, Any]] = []
    for index, item in enumerate(result.get("segments") or []):
        if not isinstance(item, dict):
            continue
        start = float(item.get("start", item.get("start_s", 0.0)))
        end = float(item.get("end", item.get("end_s", start)))
        normalized_start = max(0.0, start)
        segment: dict[str, Any] = {
            "index": index,
            "start": round(normalized_start, 3),
            "end": round(max(normalized_start, end), 3),
            "text": str(item.get("text", item.get("sentence", ""))).strip(),
        }
        for key in (
            "speaker",
            "raw_speaker",
            "speaker_label_source",
            "confidence",
            "avg_logprob",
            "no_speech_prob",
        ):
            if item.get(key) is not None:
                segment[key] = item[key]
        segments.append(segment)
    if not text and segments:
        text = "".join(str(item["text"]) for item in segments)
    return {
        "schema_version": SCHEMA_VERSION,
        "backend_id": backend_id,
        "model": model,
        "chunk_id": str(chunk["chunk_id"]),
        "audio_start_s": float(chunk["audio_start_s"]),
        "language": str(result.get("language", "ja")),
        "capabilities": sorted(set(result.get("capabilities") or ["text"])),
        "text": text,
        "segments": segments,
        "selected_term_ids": selected_term_ids,
        "context_snapshot_id": context_id,
        "metadata": result.get("metadata") or {},
    }


Transcriber = Callable[..., dict[str, Any]]


def run_backend_jobs(
    *,
    run_dir: Path,
    backend_id: str,
    model: str,
    transcriber: Transcriber,
    context_path: Path | None,
    audio_variant: str = "enhanced",
    seed_backend: str | None = None,
    top_k_terms: int = 30,
    force: bool = False,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    """Run one already-loaded backend across every transport chunk.

    Each backend lives in its own virtual environment.  The stable JSON boundary
    deliberately prevents Qwen/MOSS/Whisper dependency constraints from leaking
    into the orchestrator.
    """

    run_dir = run_dir.resolve()
    chunks = read_jsonl(run_dir / "chunks.jsonl")
    context = load_context(context_path)
    context_id = context_snapshot_id(context)
    failures: list[dict[str, str]] = []
    completed = 0
    previous_tail = ""

    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        output = result_path(run_dir, backend_id, chunk_id)
        if output.exists() and not force:
            previous_tail = str(read_json(output).get("text", ""))[-600:]
            completed += 1
            continue
        seed = (
            load_backend_result(run_dir, seed_backend, chunk_id)
            if seed_backend
            else None
        )
        seed_text = str(seed.get("text", "")) if seed else ""
        selected = select_terms(context, seed_text, top_k=top_k_terms)
        prompt = build_transcription_prompt(
            context, selected, previous_tail=previous_tail
        )
        hotwords = [str(item["canonical"]) for item in selected]
        relative_audio = (
            chunk["raw_audio"]
            if audio_variant == "raw"
            else chunk["enhanced_audio"]
        )
        audio_path = run_dir / str(relative_audio)
        try:
            kwargs = {
                "audio_path": audio_path,
                "prompt": prompt,
                "hotwords": hotwords,
                "chunk": chunk,
            }
            # Convenient for small third-party adapters that accept a subset.
            signature = inspect.signature(transcriber)
            accepted = {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters
            }
            raw_result = transcriber(**accepted)
            normalized = normalize_result(
                backend_id=backend_id,
                model=model,
                chunk=chunk,
                result=raw_result,
                selected_term_ids=[str(item["term_id"]) for item in selected],
                context_id=context_id,
            )
            write_json_atomic(output, normalized)
            previous_tail = normalized["text"][-600:]
            completed += 1
        except Exception as exc:
            failures.append({"chunk_id": chunk_id, "error": str(exc)})
            if not continue_on_error:
                raise

    summary = {
        "schema_version": SCHEMA_VERSION,
        "backend_id": backend_id,
        "model": model,
        "chunks_total": len(chunks),
        "chunks_completed": completed,
        "failures": failures,
        "context_snapshot_id": context_id,
        "audio_variant": audio_variant,
        "seed_backend": seed_backend,
    }
    write_json_atomic(run_dir / "hypotheses" / backend_id / "backend.json", summary)
    return summary

