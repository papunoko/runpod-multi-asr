#!/usr/bin/env bash
# Execute a meeting-pipeline backend using its isolated RunPod environment.

set -Eeuo pipefail
IFS=$'\n\t'

trap 'printf "ERROR: backend launch failed at line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
WORKSPACE_ROOT="${RUNPOD_WORKSPACE:-/workspace}"
ENV_ROOT="${ASR_ENV_ROOT:-/opt/venvs}"
CACHE_ROOT="${ASR_CACHE_ROOT:-$WORKSPACE_ROOT/multi-asr/cache}"

usage() {
  cat <<'EOF'
Usage: deploy/run_backend.sh BACKEND [runner arguments...]

BACKEND is one of:
  whisper  qwen  reazon  core

Examples:
  deploy/run_backend.sh whisper --run-dir /workspace/runs/demo --context context.json
  deploy/run_backend.sh qwen --run-dir /workspace/runs/demo --context context.json
  deploy/run_backend.sh reazon --run-dir /workspace/runs/demo
  deploy/run_backend.sh core prepare /workspace/jobs/demo/audio.mp3 --run-dir /workspace/jobs/demo/run
EOF
}

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
backend="$1"
shift

case "$backend" in
  whisper) env_name=whisper; module_name=meeting_pipeline.whisper_runner ;;
  qwen) env_name=qwen; module_name=meeting_pipeline.qwen3_runner ;;
  reazon) env_name=reazon; module_name=meeting_pipeline.reazon_runner ;;
  core) env_name=whisper; module_name=meeting_pipeline.cli ;;
  -h|--help) usage; exit 0 ;;
  *)
    printf 'Unsupported backend: %s\n' "$backend" >&2
    usage >&2
    exit 2
    ;;
esac

python_path="$ENV_ROOT/$env_name/bin/python"
[[ -x "$python_path" ]] || {
  printf 'Immutable image environment is missing: %s\n' "$env_name" >&2
  exit 1
}

export HF_HOME="$CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT/src"

if [[ "$backend" != core ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || {
    printf '%s\n' 'WARNING: nvidia-smi is unavailable; GPU inference may fail.' >&2
  }
fi

exec "$python_path" -B -m "$module_name" "$@"
