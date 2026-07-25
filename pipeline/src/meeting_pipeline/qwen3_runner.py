from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .backend_common import run_backend_jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-ASR isolated runner")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--device", default=os.getenv("ASR_DEVICE", "cuda:0"))
    parser.add_argument("--align", action="store_true")
    parser.add_argument("--aligner", default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--seed-backend")
    parser.add_argument("--top-k-terms", type=int, default=30)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help=(
            "Per-chunk generation ceiling. Six-minute Japanese meeting chunks "
            "normally fit below 2048; a finite ceiling limits repetition/hallucination."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise SystemExit(
            "Install this runner in its own environment: pip install -U qwen-asr"
        ) from exc

    is_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    dtype = torch.bfloat16 if is_cuda else torch.float32
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": args.device if is_cuda else "cpu",
        "max_inference_batch_size": 1,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.align:
        kwargs.update(
            {
                "forced_aligner": args.aligner,
                "forced_aligner_kwargs": {
                    "dtype": dtype,
                    "device_map": args.device if is_cuda else "cpu",
                },
            }
        )
    model = Qwen3ASRModel.from_pretrained(args.model, **kwargs)

    def transcribe(audio_path: Path, prompt: str) -> dict[str, Any]:
        result = model.transcribe(
            audio=str(audio_path),
            context=prompt,
            language="Japanese",
            return_time_stamps=args.align,
        )[0]
        segments = []
        for item in getattr(result, "time_stamps", None) or []:
            segments.append(
                {
                    "start": item.start_time,
                    "end": item.end_time,
                    "text": item.text,
                }
            )
        return {
            "language": result.language or "ja",
            "text": result.text,
            "segments": segments,
            "capabilities": ["text", "context"]
            + (["forced_alignment"] if args.align else []),
        }

    summary = run_backend_jobs(
        run_dir=args.run_dir,
        backend_id="qwen3",
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

