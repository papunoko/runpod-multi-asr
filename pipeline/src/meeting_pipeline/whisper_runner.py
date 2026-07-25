from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any

from .backend_common import run_backend_jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="faster-whisper isolated runner")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default=os.getenv("ASR_DEVICE", "cuda"))
    parser.add_argument("--compute-type", default=os.getenv("WHISPER_COMPUTE_TYPE", "float16"))
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--seed-backend")
    parser.add_argument("--top-k-terms", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit(
            "Install this runner in its own environment: pip install -U faster-whisper"
        ) from exc
    model = WhisperModel(
        args.model, device=args.device, compute_type=args.compute_type
    )

    def transcribe(
        audio_path: Path, prompt: str, hotwords: list[str]
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "language": "ja",
            "beam_size": args.beam_size,
            "vad_filter": True,
            "word_timestamps": True,
            "initial_prompt": prompt,
            "condition_on_previous_text": False,
        }
        if "hotwords" in inspect.signature(model.transcribe).parameters:
            kwargs["hotwords"] = ", ".join(hotwords)
        segment_iter, info = model.transcribe(str(audio_path), **kwargs)
        segments = []
        for item in segment_iter:
            segments.append(
                {
                    "start": item.start,
                    "end": item.end,
                    "text": item.text,
                    "avg_logprob": item.avg_logprob,
                    "no_speech_prob": item.no_speech_prob,
                }
            )
        return {
            "language": getattr(info, "language", "ja"),
            "text": "".join(item["text"] for item in segments),
            "segments": segments,
            "capabilities": ["text", "timestamps", "context", "confidence"],
            "metadata": {
                "language_probability": getattr(info, "language_probability", None)
            },
        }

    summary = run_backend_jobs(
        run_dir=args.run_dir,
        backend_id="faster_whisper",
        model=args.model,
        transcriber=transcribe,
        context_path=args.context,
        seed_backend=args.seed_backend,
        top_k_terms=args.top_k_terms,
        force=args.force,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

