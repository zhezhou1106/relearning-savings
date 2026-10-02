#!/usr/bin/env bash
# Fresh-pool relearn on existing two-arm dense_early checkpoints.
# Seeds × complements × stream steps {50,100,150}, then fraction-of-naive analysis.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export FACTTRACE_ROOT="$ROOT"
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-10}"

CONFIG="configs/experiment/phase1.yaml"
CONFIG_3ARM="configs/experiment/phase1_3arm.yaml"
SEEDS="0,1,2"
STREAM_STEPS="50,100,150"
PHASE_B_TAG="dense_early"
RELEARN_TAG="dense_early_slow1e5_fresh"
RELEARN_CONFIG="configs/train/relearn.yaml"
FORCE=0
ANALYZE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --stream-steps) STREAM_STEPS="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --no-analyze) ANALYZE=0; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

force=()
[[ "$FORCE" -eq 1 ]] && force+=(--force)

IFS=',' read -ra SEED_ARR <<< "$SEEDS"
IFS=',' read -ra STEP_ARR <<< "$STREAM_STEPS"

for seed in "${SEED_ARR[@]}"; do
  for c in 0 1; do
    for step in "${STEP_ARR[@]}"; do
      echo "=== fresh relearn seed=$seed c=$c stream=$step ==="
      uv run python scripts/relearn_sweep.py \
        --config "$CONFIG" \
        --seed "$seed" \
        --complementary "$c" \
        --tag "$RELEARN_TAG" \
        --phase-b-tag "$PHASE_B_TAG" \
        --relearn-tag "$RELEARN_TAG" \
        --relearn-config "$RELEARN_CONFIG" \
        --stream-steps "$step" \
        --fact-source fresh_pool \
        --fresh-pool data/facts/fresh_pool.parquet \
        --split confirmatory \
        "${force[@]}"
    done
  done
done

if [[ "$ANALYZE" -eq 1 ]]; then
  for seed in "${SEED_ARR[@]}"; do
    echo "=== fraction of naive (2-arm) seed=$seed ==="
    uv run python scripts/analyze_fraction_of_naive.py \
      --config "$CONFIG" --seed "$seed" "${force[@]}"
    if [[ -f "$CONFIG_3ARM" ]]; then
      echo "=== fraction of naive (3-arm) seed=$seed ==="
      uv run python scripts/analyze_fraction_of_naive.py \
        --config "$CONFIG" \
        --panel-config "$CONFIG_3ARM" \
        --seed "$seed" \
        --contrasts learned:control,learned:wrong,wrong:control \
        "${force[@]}"
    fi
  done
fi

echo "Fresh-pool relearn complete: seeds=$SEEDS steps=$STREAM_STEPS"
