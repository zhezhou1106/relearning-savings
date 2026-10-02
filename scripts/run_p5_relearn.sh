#!/usr/bin/env bash
# P5 relearning sweep: savings vs forgetting at multiple Phase B checkpoints.
#
# Stages:
#   1. (optional) dense_early Phase B for seeds missing checkpoints
#   2. Relearning at each stream checkpoint on dev or confirmatory panel
#   3. Merge curves + savings-vs-forgetting analysis
#
# Usage:
#   ./scripts/run_p5_relearn.sh --split confirmatory --seeds 0,1,2 --prepare-phase-b
#   ./scripts/run_p5_relearn.sh --config configs/experiment/phase1_3arm.yaml --split confirmatory --seeds 0,1,2
#   ./scripts/run_p5_relearn.sh --split confirmatory --seeds 0 --analyze-only
# Defaults match the locked confirmatory protocol (lr=1e-5, tag dense_early_slow1e5).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export FACTTRACE_ROOT="$ROOT"
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-10}"

CONFIG="configs/experiment/phase1.yaml"
PHASE_B_TAG="dense_early"
PHASE_B_CFG="configs/train/phase_b_dense_early.yaml"
BEH_CFG="configs/eval/behavioral_pilot.yaml"
RELEARN_TAG="dense_early_slow1e5"
RELEARN_CONFIG="configs/train/relearn.yaml"
SPLIT="confirmatory"
SEEDS="0,1,2"
STREAM_STEPS="0,25,50,75,100,125,150,200,250,400"
CONTRASTS=""
MAX_FACTS=""
PREPARE_PHASE_B=0
ANALYZE_ONLY=0
FORCE=0
MERGE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --tag) PHASE_B_TAG="$2"; shift 2 ;;
    --phase-b-tag) PHASE_B_TAG="$2"; shift 2 ;;
    --relearn-tag) RELEARN_TAG="$2"; shift 2 ;;
    --relearn-config) RELEARN_CONFIG="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --stream-steps) STREAM_STEPS="$2"; shift 2 ;;
    --contrasts) CONTRASTS="$2"; shift 2 ;;
    --max-facts) MAX_FACTS="$2"; shift 2 ;;
    --prepare-phase-b) PREPARE_PHASE_B=1; shift ;;
    --analyze-only) ANALYZE_ONLY=1; shift ;;
    --force) FORCE=1; shift ;;
    --no-merge) MERGE=0; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

EXP="$(awk '/^experiment_name:/{print $2; exit}' "$CONFIG")"
N_RUNS="$(awk '/^n_runs:/{print $2; exit}' "$CONFIG")"
N_RUNS="${N_RUNS:-2}"
if [[ -z "$CONTRASTS" ]]; then
  if [[ "$N_RUNS" -ge 3 ]]; then
    CONTRASTS="learned:control,learned:wrong,wrong:control"
  else
    CONTRASTS="learned:control"
  fi
fi

force=()
[[ "$FORCE" -eq 1 ]] && force+=(--force)
extra=()
[[ -n "$MAX_FACTS" ]] && extra+=(--max-facts "$MAX_FACTS")
relearn_cfg=()
[[ -n "$RELEARN_CONFIG" ]] && relearn_cfg+=(--relearn-config "$RELEARN_CONFIG")
contrast_args=(--contrasts "$CONTRASTS")

IFS=',' read -ra SEED_ARR <<< "$SEEDS"

if [[ "$ANALYZE_ONLY" -eq 1 ]]; then
  for seed in "${SEED_ARR[@]}"; do
    uv run python scripts/analyze_savings_vs_forgetting.py \
      --config "$CONFIG" --seed "$seed" --tag "$RELEARN_TAG" \
      --phase-b-tag "$PHASE_B_TAG" --relearn-tag "$RELEARN_TAG" \
      --split "$SPLIT" --stream-steps "$STREAM_STEPS" \
      "${contrast_args[@]}" "${force[@]}"
  done
  exit 0
fi

for seed in "${SEED_ARR[@]}"; do
  if [[ "$PREPARE_PHASE_B" -eq 1 ]]; then
    missing=0
    for step in ${STREAM_STEPS//,/ }; do
      if [[ ! -d "artifacts/runs/${EXP}/seed_${seed}/c0/phase_b_${PHASE_B_TAG}/ckpt_${step}" ]]; then
        missing=1
        break
      fi
    done
    if [[ "$missing" -eq 1 ]]; then
      echo "=== P5 prep: Phase B tag=$PHASE_B_TAG exp=$EXP seed=$seed runs=$N_RUNS ==="
      for ((c = 0; c < N_RUNS; c++)); do
        uv run python scripts/train_phase_b.py \
          --config "$CONFIG" --seed "$seed" --complementary "$c" \
          --phase-b-config "$PHASE_B_CFG" --tag "$PHASE_B_TAG" "${force[@]}"
      done
      for ((c = 0; c < N_RUNS; c++)); do
        uv run python scripts/eval_retention.py \
          --config "$CONFIG" --seed "$seed" --complementary "$c" \
          --phase-b-config "$PHASE_B_CFG" --tag "$PHASE_B_TAG" "${force[@]}"
      done
      uv run python scripts/select_checkpoints.py \
        --config "$CONFIG" --seed "$seed" \
        --phase-b-config "$PHASE_B_CFG" \
        --behavioral-config "$BEH_CFG" \
        --tag "$PHASE_B_TAG" "${force[@]}"
    fi
  fi

  echo "=== P5 relearn split=$SPLIT exp=$EXP seed=$seed runs=$N_RUNS phase_b=$PHASE_B_TAG relearn=$RELEARN_TAG ==="
  for ((c = 0; c < N_RUNS; c++)); do
    uv run python scripts/relearn_sweep.py \
      --config "$CONFIG" --seed "$seed" --complementary "$c" \
      --tag "$RELEARN_TAG" --phase-b-tag "$PHASE_B_TAG" --relearn-tag "$RELEARN_TAG" \
      --stream-steps "$STREAM_STEPS" --split "$SPLIT" \
      "${relearn_cfg[@]}" "${extra[@]}" "${force[@]}"
  done
  if [[ "$MERGE" -eq 1 ]]; then
    uv run python scripts/relearn_sweep.py \
      --config "$CONFIG" --seed "$seed" --complementary 0 \
      --tag "$RELEARN_TAG" --relearn-tag "$RELEARN_TAG" \
      --stream-steps "$STREAM_STEPS" --split "$SPLIT" --merge-only
  fi
  uv run python scripts/analyze_savings_vs_forgetting.py \
    --config "$CONFIG" --seed "$seed" --tag "$RELEARN_TAG" \
    --phase-b-tag "$PHASE_B_TAG" --relearn-tag "$RELEARN_TAG" \
    --split "$SPLIT" --stream-steps "$STREAM_STEPS" \
    "${contrast_args[@]}" "${force[@]}"
done

if [[ "${#SEED_ARR[@]}" -gt 1 ]]; then
  first="${SEED_ARR[0]}"
  pooled_json="artifacts/analysis/p5/savings_vs_forgetting_seed_${first}_${RELEARN_TAG}_${SPLIT}.json"
  if [[ -f "$pooled_json" ]]; then
    echo "=== Pooled savings-vs-stream plot ==="
    uv run python scripts/plot_savings_vs_stream.py \
      --config "$CONFIG" --tag "$RELEARN_TAG" --split "$SPLIT" --seeds "$SEEDS"
  else
    echo "Skip pooled plot (no 2-arm analysis JSON at $pooled_json)."
  fi
fi

echo "P5 relearn complete: exp=$EXP split=$SPLIT seeds=$SEEDS phase_b=$PHASE_B_TAG relearn=$RELEARN_TAG"
