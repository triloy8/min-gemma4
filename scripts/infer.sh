#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROMPT="${PROMPT:-Hello.}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
LOADER="${LOADER:-naive}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_K="${TOP_K:-0}"
SEED="${SEED:-0}"

ARGS=(
  --prompt "$PROMPT"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --loader "$LOADER"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top-k "$TOP_K"
  --seed "$SEED"
)

exec uv run python scripts/infer.py "${ARGS[@]}" "$@"
