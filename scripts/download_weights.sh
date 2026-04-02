#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VARIANT="${VARIANT:-e4b-it}"
OUTPUT="${OUTPUT:-$ROOT_DIR/model.safetensors}"
FORCE="${FORCE:-0}"

ARGS=(
  --variant "$VARIANT"
  --output "$OUTPUT"
)

if [[ "$FORCE" == "1" ]]; then
  ARGS+=(--force)
fi

exec uv run python scripts/download_weights.py "${ARGS[@]}" "$@"
