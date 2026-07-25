from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .audio import detect_silences, prepare_audio, probe_audio
from .utils import read_json, write_json_atomic


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _known_speaker(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=/path/to/2-10s-reference.wav")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Known speaker requires both NAME and path")
    return name.strip(), _path(raw_path.strip())


def _backend_prior(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected BACKEND=WEIGHT")
    backend, raw_weight = value.split("=", 1)
    try:
        weight = float(raw_weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Backend weight must be numeric") from exc
    if not backend.strip() or weight < 0:
        raise argparse.ArgumentTypeError("Backend name is required and weight must be >= 0")
    return backend.strip(), weight


def _add_prepare_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("audio", type=_path)
    parser.add_argument("--run-dir", required=True, type=_path)
    parser.add_argument("--target-seconds", type=float, default=360.0)
    parser.add_argument("--overlap-seconds", type=float, default=8.0)
    parser.add_argument("--silence-threshold-db", type=float, default=-35.0)
    parser.add_argument("--min-silence-seconds", type=float, default=0.8)
    parser.add_argument("--search-window-seconds", type=float, default=45.0)
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Keep only high-pass filtering in the contextual-ASR WAV",
    )
    parser.add_argument("--force", action="store_true")


def _add_api_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True, type=_path)
    parser.add_argument("--context", type=_path)
    parser.add_argument(
        "--known-speaker",
        type=_known_speaker,
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="2-10 second reference; repeat up to four times",
    )
    parser.add_argument(
        "--transcribe-model",
        default=os.getenv("MEETING_TRANSCRIBE_MODEL", "gpt-4o-transcribe"),
    )
    parser.add_argument(
        "--diarize-model",
        default=os.getenv(
            "MEETING_DIARIZE_MODEL", "gpt-4o-transcribe-diarize"
        ),
    )
    parser.add_argument("--top-k-terms", type=int, default=30)
    parser.add_argument("--force", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meeting-pipeline",
        description="Backend-neutral multi-ASR Japanese meeting transcription, fusion, evaluation, and grounded minutes",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Probe audio without API calls")
    inspect_parser.add_argument("audio", type=_path)
    inspect_parser.add_argument("--output", type=_path)
    inspect_parser.add_argument("--silence-threshold-db", type=float, default=-35.0)
    inspect_parser.add_argument("--min-silence-seconds", type=float, default=0.8)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Create silence-aware API transport chunks"
    )
    _add_prepare_options(prepare_parser)

    transcribe_parser = subparsers.add_parser(
        "transcribe", help="Legacy optional OpenAI two-pass profile"
    )
    _add_api_options(transcribe_parser)

    reconcile_parser = subparsers.add_parser(
        "reconcile", help="Project contextual text onto speaker/time segments"
    )
    reconcile_parser.add_argument("--run-dir", required=True, type=_path)
    reconcile_parser.add_argument(
        "--known-speaker-name", action="append", default=[]
    )

    minutes_parser = subparsers.add_parser(
        "minutes", help="Create and verify evidence-grounded minutes"
    )
    minutes_parser.add_argument("--run-dir", required=True, type=_path)
    minutes_parser.add_argument("--context", type=_path)
    minutes_parser.add_argument(
        "--model",
        default=os.getenv("MEETING_MINUTES_MODEL", "gpt-5.6-terra"),
    )
    minutes_parser.add_argument("--force", action="store_true")

    extractive_minutes_parser = subparsers.add_parser(
        "minutes-extractive",
        help="Create a no-API verbatim review draft; semantic claims remain unverified",
    )
    extractive_minutes_parser.add_argument("--run-dir", required=True, type=_path)
    extractive_minutes_parser.add_argument("--context", type=_path)
    extractive_minutes_parser.add_argument("--max-per-kind", type=int, default=25)
    extractive_minutes_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser(
        "run", help="Legacy optional OpenAI profile; use isolated runners + fuse for broad comparison"
    )
    _add_prepare_options(run_parser)
    run_parser.add_argument("--context", type=_path)
    run_parser.add_argument(
        "--known-speaker",
        type=_known_speaker,
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    run_parser.add_argument(
        "--transcribe-model",
        default=os.getenv("MEETING_TRANSCRIBE_MODEL", "gpt-4o-transcribe"),
    )
    run_parser.add_argument(
        "--diarize-model",
        default=os.getenv(
            "MEETING_DIARIZE_MODEL", "gpt-4o-transcribe-diarize"
        ),
    )
    run_parser.add_argument(
        "--minutes-model",
        default=os.getenv("MEETING_MINUTES_MODEL", "gpt-5.6-terra"),
    )
    run_parser.add_argument("--top-k-terms", type=int, default=30)
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare chunks and manifest, but make no API calls",
    )
    run_parser.add_argument("--skip-minutes", action="store_true")

    subparsers.add_parser(
        "backends", help="Show the comparison matrix and isolated runner commands"
    )

    fuse_parser = subparsers.add_parser(
        "fuse", help="Project multiple ASR hypotheses to a speaker/time anchor and fuse"
    )
    fuse_parser.add_argument("--run-dir", required=True, type=_path)
    fuse_parser.add_argument("--anchor", required=True)
    fuse_parser.add_argument("--candidate", action="append", default=[])
    fuse_parser.add_argument(
        "--phonetic-evidence",
        action="append",
        default=[],
        help="Kana/phoneme backend used only to support or reject glossary terms",
    )
    fuse_parser.add_argument("--context", type=_path)
    fuse_parser.add_argument(
        "--backend-prior", type=_backend_prior, action="append", default=[]
    )
    fuse_parser.add_argument("--minimum-projection-ratio", type=float, default=0.35)
    fuse_parser.add_argument("--review-disagreement-threshold", type=float, default=0.25)
    fuse_parser.add_argument("--use-pyopenjtalk", action="store_true")

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Score a transcript against human reference JSONL"
    )
    evaluate_parser.add_argument("--reference", required=True, type=_path)
    evaluate_parser.add_argument("--hypothesis", required=True, type=_path)
    evaluate_parser.add_argument("--context", type=_path)
    evaluate_parser.add_argument("--output", type=_path)

    comparison_parser = subparsers.add_parser(
        "comparison-report",
        help="Create gold-free backend disagreement diagnostics from fusion audit",
    )
    comparison_parser.add_argument("--run-dir", required=True, type=_path)
    comparison_parser.add_argument("--output-json", type=_path)
    comparison_parser.add_argument("--output-markdown", type=_path)
    comparison_parser.add_argument("--max-review-segments", type=int, default=30)

    validation_parser = subparsers.add_parser(
        "validate-backend",
        help="Validate timestamped chunk results before using a backend as fusion anchor",
    )
    validation_parser.add_argument("--run-dir", required=True, type=_path)
    validation_parser.add_argument("--backend", required=True)
    validation_parser.add_argument("--output", type=_path)
    validation_parser.add_argument("--minimum-temporal-span-ratio", type=float, default=0.75)
    validation_parser.add_argument("--timestamp-tolerance-seconds", type=float, default=1.0)

    annotation_parser = subparsers.add_parser(
        "annotation-template", help="Create stratified intervals for human gold transcription"
    )
    annotation_parser.add_argument("--run-dir", required=True, type=_path)
    annotation_parser.add_argument("--total-minutes", type=float, default=15.0)
    annotation_parser.add_argument("--window-seconds", type=float, default=90.0)
    annotation_parser.add_argument("--output", type=_path)

    import_parser = subparsers.add_parser(
        "import-hypotheses",
        help="Import PLAUD TXT or other JSON/JSONL output as a candidate",
    )
    import_parser.add_argument("--run-dir", required=True, type=_path)
    import_parser.add_argument("--input", required=True, type=_path)
    import_parser.add_argument("--backend", required=True)
    import_parser.add_argument("--model", default="external")
    import_parser.add_argument("--force", action="store_true")

    retrieval_parser = subparsers.add_parser(
        "retrieve-context",
        help="Build/augment context.json from offline JSON/JSONL term catalogs",
    )
    retrieval_parser.add_argument(
        "--catalog",
        required=True,
        type=_path,
        action="append",
        help="Term catalog; repeat for multiple JSON/JSONL files",
    )
    retrieval_parser.add_argument("--output-context", required=True, type=_path)
    retrieval_parser.add_argument("--base-context", type=_path)
    retrieval_parser.add_argument("--meeting-id")
    retrieval_parser.add_argument("--title")
    retrieval_parser.add_argument(
        "--agenda", action="append", default=[], help="Agenda item; repeat as needed"
    )
    retrieval_parser.add_argument(
        "--seed-transcript",
        type=_path,
        help="Optional TXT/MD/JSON/JSONL ranking signal; never a canonical term source",
    )
    retrieval_parser.add_argument("--seed-max-chars", type=int, default=200_000)
    retrieval_parser.add_argument("--top-k", type=int, default=30)
    retrieval_parser.add_argument("--minimum-score", type=float, default=0.001)
    retrieval_parser.add_argument("--audit", type=_path)
    retrieval_parser.add_argument("--force", action="store_true")
    return parser


def _inspect(args: argparse.Namespace) -> dict[str, Any]:
    probe = probe_audio(args.audio.resolve())
    silences = detect_silences(
        args.audio.resolve(),
        threshold_db=args.silence_threshold_db,
        min_silence_s=args.min_silence_seconds,
    )
    payload = {
        "probe": probe,
        "silence_analysis": {
            "threshold_db": args.silence_threshold_db,
            "min_silence_s": args.min_silence_seconds,
            "count": len(silences),
            "total_s": round(sum(item.duration_s for item in silences), 3),
            "max_s": round(max((item.duration_s for item in silences), default=0.0), 3),
        },
    }
    if args.output:
        write_json_atomic(args.output, payload)
    return payload


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    return prepare_audio(
        args.audio,
        args.run_dir,
        target_chunk_s=args.target_seconds,
        overlap_s=args.overlap_seconds,
        silence_threshold_db=args.silence_threshold_db,
        min_silence_s=args.min_silence_seconds,
        search_window_s=args.search_window_seconds,
        normalize=not args.no_normalize,
        force=args.force,
    )


def _transcribe(args: argparse.Namespace) -> dict[str, Any]:
    from .openai_steps import transcribe_run

    return transcribe_run(
        args.run_dir,
        context_path=args.context,
        known_speakers=args.known_speaker,
        transcribe_model=args.transcribe_model,
        diarize_model=args.diarize_model,
        top_k_terms=args.top_k_terms,
        force=args.force,
    )


def _reconcile(args: argparse.Namespace) -> dict[str, Any]:
    from .reconcile import reconcile_run

    names = set(args.known_speaker_name)
    models_path = args.run_dir / "api_models.json"
    if models_path.exists():
        names.update(read_json(models_path).get("known_speaker_names", []))
    segments = reconcile_run(args.run_dir, known_speaker_names=names)
    return {"segments": len(segments), "output": str(args.run_dir / "transcript.md")}


def _minutes(args: argparse.Namespace) -> dict[str, Any]:
    from .minutes import generate_minutes

    return generate_minutes(
        args.run_dir,
        context_path=args.context,
        model=args.model,
        force=args.force,
    )


def _minutes_extractive(args: argparse.Namespace) -> dict[str, Any]:
    from .minutes import generate_extractive_minutes

    return generate_extractive_minutes(
        args.run_dir,
        context_path=args.context,
        max_per_kind=args.max_per_kind,
        force=args.force,
    )


def _backends() -> dict[str, Any]:
    return {
        "principle": "Keep error-diverse systems; select by Japanese meeting gold data, not public leaderboard rank.",
        "runners": [
            {"id": "faster_whisper", "family": "Whisper encoder-decoder", "command": "meeting-whisper"},
            {"id": "kotoba_whisper", "family": "Japanese Whisper", "command": "meeting-kotoba"},
            {"id": "reazon_nemo", "family": "Japanese FastConformer RNN-T", "command": "meeting-reazon"},
            {"id": "qwen3", "family": "contextual Speech-LLM", "command": "meeting-qwen3"},
            {"id": "funasr_nano", "family": "hotword Speech-LLM", "command": "meeting-funasr"},
            {"id": "moss", "family": "joint ASR + diarization", "command": "meeting-moss"},
            {"id": "ctc", "family": "kana/phoneme or character CTC acoustic witness", "command": "meeting-ctc --model MODEL_ID"},
            {"id": "openai_content/openai_diarize", "family": "optional API challenger", "command": "meeting-openai --mode content|diarize"},
        ],
        "fusion": "speaker/time anchor -> per-segment projection -> weighted MBR over lexical ASR only -> kana/phoneme-guarded term correction -> review queue",
    }


def _fuse(args: argparse.Namespace) -> dict[str, Any]:
    from .ensemble import fuse_run

    return fuse_run(
        args.run_dir,
        anchor_backend=args.anchor,
        candidate_backends=args.candidate,
        phonetic_evidence_backends=args.phonetic_evidence,
        context_path=args.context,
        backend_priors=dict(args.backend_prior),
        minimum_projection_ratio=args.minimum_projection_ratio,
        review_disagreement_threshold=args.review_disagreement_threshold,
        use_pyopenjtalk=args.use_pyopenjtalk,
    )


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from .evaluate import evaluate_files

    return evaluate_files(
        args.reference, args.hypothesis, args.context, args.output
    )


def _comparison_report(args: argparse.Namespace) -> dict[str, Any]:
    from .comparison import generate_comparison_report

    return generate_comparison_report(
        args.run_dir,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        max_review_segments=args.max_review_segments,
    )


def _validate_backend(args: argparse.Namespace) -> dict[str, Any]:
    from .backend_validation import validate_timestamped_backend

    return validate_timestamped_backend(
        args.run_dir,
        backend_id=args.backend,
        output=args.output,
        minimum_temporal_span_ratio=args.minimum_temporal_span_ratio,
        timestamp_tolerance_s=args.timestamp_tolerance_seconds,
    )


def _annotation_template(args: argparse.Namespace) -> dict[str, Any]:
    from .annotation import create_annotation_template

    return create_annotation_template(
        args.run_dir,
        total_minutes=args.total_minutes,
        window_seconds=args.window_seconds,
        output=args.output,
    )


def _import_hypotheses(args: argparse.Namespace) -> dict[str, Any]:
    from .importer import import_hypotheses

    return import_hypotheses(
        args.run_dir,
        input_path=args.input,
        backend_id=args.backend,
        model=args.model,
        force=args.force,
    )


def _retrieve_context(args: argparse.Namespace) -> dict[str, Any]:
    from .retrieval import retrieve_context

    return retrieve_context(
        catalog_paths=args.catalog,
        output_context=args.output_context,
        base_context_path=args.base_context,
        meeting_id=args.meeting_id,
        title=args.title,
        agenda=args.agenda,
        seed_transcript_path=args.seed_transcript,
        seed_max_chars=args.seed_max_chars,
        top_k=args.top_k,
        minimum_score=args.minimum_score,
        audit_path=args.audit,
        force=args.force,
    )


def _run_all(args: argparse.Namespace) -> dict[str, Any]:
    prepared = _prepare(args)
    if args.dry_run:
        return {
            "status": "prepared_dry_run",
            "chunk_count": prepared["chunk_count"],
            "run_dir": str(args.run_dir),
        }
    from .openai_steps import transcribe_run
    from .reconcile import reconcile_run

    transcribed = transcribe_run(
        args.run_dir,
        context_path=args.context,
        known_speakers=args.known_speaker,
        transcribe_model=args.transcribe_model,
        diarize_model=args.diarize_model,
        top_k_terms=args.top_k_terms,
        force=args.force,
    )
    names = {name for name, _ in args.known_speaker}
    segments = reconcile_run(args.run_dir, known_speaker_names=names)
    result: dict[str, Any] = {
        "status": "transcribed",
        "chunks": transcribed["chunks"],
        "segments": len(segments),
        "transcript": str(args.run_dir / "transcript.md"),
    }
    if not args.skip_minutes:
        from .minutes import generate_minutes

        generate_minutes(
            args.run_dir,
            context_path=args.context,
            model=args.minutes_model,
            force=args.force,
        )
        result["status"] = "completed"
        result["minutes"] = str(args.run_dir / "minutes.md")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = _inspect(args)
        elif args.command == "prepare":
            result = _prepare(args)
        elif args.command == "transcribe":
            result = _transcribe(args)
        elif args.command == "reconcile":
            result = _reconcile(args)
        elif args.command == "minutes":
            result = _minutes(args)
        elif args.command == "minutes-extractive":
            result = _minutes_extractive(args)
        elif args.command == "run":
            result = _run_all(args)
        elif args.command == "backends":
            result = _backends()
        elif args.command == "fuse":
            result = _fuse(args)
        elif args.command == "evaluate":
            result = _evaluate(args)
        elif args.command == "comparison-report":
            result = _comparison_report(args)
        elif args.command == "validate-backend":
            result = _validate_backend(args)
        elif args.command == "annotation-template":
            result = _annotation_template(args)
        elif args.command == "import-hypotheses":
            result = _import_hypotheses(args)
        elif args.command == "retrieve-context":
            result = _retrieve_context(args)
        else:
            parser.error(f"Unknown command: {args.command}")
            return 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

