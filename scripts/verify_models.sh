#!/usr/bin/env bash
# Verify locally present model snapshots without downloading or modifying them.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
MODEL_ROOT="${1:-$REPO_ROOT/models}"

case "$MODEL_ROOT" in
  /*) ;;
  *) MODEL_ROOT="$PWD/$MODEL_ROOT" ;;
esac
if [[ ! -d "$MODEL_ROOT" || -L "$MODEL_ROOT" ]]; then
  printf 'Model root is missing or unsafe: %s\n' "$MODEL_ROOT" >&2
  exit 2
fi

TEMP_ROOT="$(mktemp -d)"
trap 'rm -rf -- "$TEMP_ROOT"' EXIT

verify_one() {
  local name="$1"
  local destination="$MODEL_ROOT/$name"
  local manifest="$REPO_ROOT/model-manifests/$name.sha256"
  if [[ ! -d "$destination" || -L "$destination" ]]; then
    printf 'Model destination is missing or unsafe: %s\n' "$destination" >&2
    return 2
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

verify_one whisper
verify_one reazon
verify_one qwen
printf 'All three model snapshots match their exact file manifests.\n'
