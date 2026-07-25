from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .backend_common import run_backend_jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ReazonSpeech NeMo v2 Japanese RNN-T isolated runner"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--model", default="reazon-research/reazonspeech-nemo-v2")
    parser.add_argument("--device", default=os.getenv("ASR_DEVICE", "cuda"))
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as exc:
        raise SystemExit(
            "Install NVIDIA NeMo ASR in a dedicated environment for this runner."
        ) from exc
    model_path = args.model.resolve() if isinstance(args.model, Path) else Path(args.model)
    if model_path.is_file():
        model = nemo_asr.models.ASRModel.restore_from(str(model_path))
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(str(args.model))
    if hasattr(model, "to"):
        model = model.to(args.device)
    model.eval()

    def transcribe(audio_path: Path) -> dict[str, Any]:
        raw = model.transcribe([str(audio_path)])
        first = raw[0]
        text = first.text if hasattr(first, "text") else str(first)
        return {
            "text": text,
            "segments": [],
            "capabilities": ["text", "japanese_rnnt", "long_form"],
            "metadata": {
                "context_used": False,
                "role": "architecturally independent Japanese witness",
            },
        }

    summary = run_backend_jobs(
        run_dir=args.run_dir,
        backend_id="reazon_nemo",
        model=args.model,
        transcriber=transcribe,
        context_path=args.context,
        top_k_terms=0,
        force=args.force,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
