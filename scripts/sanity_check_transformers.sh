#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROMPT="${PROMPT:-Hello.}"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-bfloat16}"
LOADER="${LOADER:-naive}"
TOP_K="${TOP_K:-5}"
LAYERWISE="${LAYERWISE:-1}"

ARGS=(
  --prompt "$PROMPT"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --loader "$LOADER"
  --top-k "$TOP_K"
)

if [[ "$LAYERWISE" == "1" ]]; then
  ARGS+=(--layerwise)
fi

exec uv run python scripts/sanity_check_transformers.py "${ARGS[@]}" "$@"
