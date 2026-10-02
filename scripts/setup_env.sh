#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# Always bind to this checkout. Do not inherit a parent-shell FACTTRACE_ROOT.
export FACTTRACE_ROOT="$ROOT"
export HF_HOME="${HF_HOME:-$ROOT/artifacts/hf_cache}"
mkdir -p "$HF_HOME" artifacts data/facts data/templates data/manifests data/manifests_3arm
uv sync
echo "Environment ready."
echo "  FACTTRACE_ROOT=$FACTTRACE_ROOT"
echo "  HF_HOME=$HF_HOME"
echo "Optional: export WANDB_API_KEY=... and FACTTRACE_WANDB=1 for W&B logging."
