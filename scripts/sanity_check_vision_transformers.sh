#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IMAGE="${IMAGE:-}"
PROMPT="${PROMPT:-Describe the image briefly.}"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-bfloat16}"
LOADER="${LOADER:-naive}"
TOP_K="${TOP_K:-5}"

ARGS=(
  --prompt "$PROMPT"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --loader "$LOADER"
  --top-k "$TOP_K"
)

if [[ -n "$IMAGE" ]]; then
  ARGS+=(--image "$IMAGE")
fi

exec uv run python scripts/sanity_check_vision_transformers.py "${ARGS[@]}" "$@"
