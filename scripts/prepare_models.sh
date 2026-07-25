#!/usr/bin/env bash
# Download the three immutable Hugging Face snapshots used by the OCI image.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
MODEL_ROOT="${1:-$REPO_ROOT/models}"

case "$MODEL_ROOT" in
  /*) ;;
  *) MODEL_ROOT="$PWD/$MODEL_ROOT" ;;
esac
case "$MODEL_ROOT" in
  /|/opt|/workspace|"$REPO_ROOT")
    printf 'Unsafe model root: %s\n' "$MODEL_ROOT" >&2
    exit 2
    ;;
esac
if [[ -L "$MODEL_ROOT" ]]; then
  printf 'Model root must not be a symlink: %s\n' "$MODEL_ROOT" >&2
  exit 2
fi
mkdir -p -- "$MODEL_ROOT"
TEMP_ROOT="$(mktemp -d)"
trap 'rm -rf -- "$TEMP_ROOT"' EXIT

download_and_verify() {
  local name="$1"
  local repository="$2"
  local revision="$3"
  local destination="$MODEL_ROOT/$name"
  local manifest="$REPO_ROOT/model-manifests/$name.sha256"
  local resolved_destination
  local resolved_cache

  if [[ -L "$destination" ]]; then
    printf 'Model destination must not be a symlink: %s\n' "$destination" >&2
    return 2
  fi
  mkdir -p -- "$destination"
  uv tool run --from huggingface-hub==0.36.2 \
    hf download "$repository" \
    --revision "$revision" \
    --local-dir "$destination"

  if [[ -d "$destination/.cache" ]]; then
    resolved_destination="$(realpath -e -- "$destination")"
    resolved_cache="$(realpath -e -- "$destination/.cache")"
    case "$resolved_cache" in
      "$resolved_destination"/.cache) rm -rf -- "$resolved_cache" ;;
      *) printf 'Refusing unsafe cache removal: %s\n' "$resolved_cache" >&2; return 2 ;;
    esac
  fi
  if find "$destination" -type l -print -quit | grep -q .; then
    printf 'Model snapshot contains a symlink: %s\n' "$destination" >&2
    return 2
  fi

  (
    cd -- "$destination"
    sha256sum --check --strict "$manifest"
    find . -type f -printf '%P\n' | LC_ALL=C sort > "$TEMP_ROOT/$name.actual"
    awk '{$1=""; sub(/^ +/, ""); print}' "$manifest" \
      | LC_ALL=C sort > "$TEMP_ROOT/$name.expected"
    cmp "$TEMP_ROOT/$name.expected" "$TEMP_ROOT/$name.actual"
  )
}

export HF_HUB_DISABLE_TELEMETRY=1
download_and_verify \
  whisper \
  Systran/faster-whisper-large-v3 \
  edaa852ec7e145841d8ffdb056a99866b5f0a478
download_and_verify \
  reazon \
  reazon-research/reazonspeech-nemo-v2 \
  33693408be76b7cba9fd4a7546a0a8772430211b
download_and_verify \
  qwen \
  Qwen/Qwen3-ASR-1.7B \
  7278e1e70fe206f11671096ffdd38061171dd6e5

printf 'All three immutable model snapshots match their manifests.\n'
