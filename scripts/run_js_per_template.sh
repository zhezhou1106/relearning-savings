#!/usr/bin/env bash
# Per-template eval on dense_early checkpoints, then JS noise-floor analysis.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export FACTTRACE_ROOT="$ROOT"
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-10}"

CONFIG="configs/experiment/phase1.yaml"
PHASE_B_CFG="configs/train/phase_b_dense_early.yaml"
TAG="dense_early"
SEEDS="0,1,2"
STEPS="0 25 50 75 100 125 150 200 250 400"
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

force=()
[[ "$FORCE" -eq 1 ]] && force+=(--force)

IFS=',' read -ra SEED_ARR <<< "$SEEDS"

for seed in "${SEED_ARR[@]}"; do
  for c in 0 1; do
    echo "=== per-template eval seed=$seed c=$c ==="
    uv run python scripts/eval_retention.py \
      --config "$CONFIG" \
      --seed "$seed" \
      --complementary "$c" \
      --phase-b-config "$PHASE_B_CFG" \
      --tag "$TAG" \
      --steps $STEPS \
      --per-template \
      "${force[@]}"
  done
done

echo "=== JS noise floor ==="
uv run python scripts/js_noise_floor.py --config "$CONFIG" --tag "$TAG" --seeds "$SEEDS" "${force[@]}"

echo "JS per-template eval complete: seeds=$SEEDS"
