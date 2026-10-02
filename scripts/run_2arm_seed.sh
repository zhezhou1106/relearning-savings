#!/usr/bin/env bash
# One 2-arm seed: Phase A → dense_early Phase B (c0/c1) → eval → nominate.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export FACTTRACE_ROOT="$ROOT"
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-10}"

CONFIG="configs/experiment/phase1.yaml"
PHASE_B_CFG="configs/train/phase_b_dense_early.yaml"
BEH_CFG="configs/eval/behavioral_pilot.yaml"
TAG="dense_early"
SEED=""
DRY_RUN=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$SEED" ]]; then
  echo "Usage: $0 --seed N [--config PATH] [--dry-run] [--force]"
  exit 1
fi

extra=()
force=()
[[ "$DRY_RUN" -eq 1 ]] && extra+=(--dry-run)
[[ "$FORCE" -eq 1 ]] && force+=(--force)

echo "=== 2-arm Phase A seed=$SEED ==="
uv run python scripts/train_phase_a.py --config "$CONFIG" --seed "$SEED" "${extra[@]}" "${force[@]}"

for c in 0 1; do
  echo "=== 2-arm Phase B tag=$TAG seed=$SEED c=$c ==="
  uv run python scripts/train_phase_b.py \
    --config "$CONFIG" --seed "$SEED" --complementary "$c" \
    --phase-b-config "$PHASE_B_CFG" --tag "$TAG" "${extra[@]}" "${force[@]}"
done

for c in 0 1; do
  echo "=== 2-arm eval tag=$TAG seed=$SEED c=$c ==="
  uv run python scripts/eval_retention.py \
    --config "$CONFIG" --seed "$SEED" --complementary "$c" \
    --phase-b-config "$PHASE_B_CFG" --tag "$TAG" "${extra[@]}" "${force[@]}"
done

echo "=== 2-arm nominate tag=$TAG seed=$SEED ==="
uv run python scripts/select_checkpoints.py \
  --config "$CONFIG" --seed "$SEED" \
  --phase-b-config "$PHASE_B_CFG" \
  --behavioral-config "$BEH_CFG" \
  --tag "$TAG" "${force[@]}"

echo "Two-arm seed $SEED complete (Phase A + dense_early Phase B + eval + nominate)."
