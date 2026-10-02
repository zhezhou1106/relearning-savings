# FactTrace

Code, data, and results for the paper *Binding-Specific Relearning Savings for
Test-Time Continual Learning* (NeurIPS 2026 TTCL Workshop): whether a previously learned factual binding is
cheaper to reacquire than a matched never-learned binding after rehearsal-free
continual fine-tuning.

The repository supports two levels of reproduction:

1. **From the released results (no GPU, about a minute).** The relearning
   curves, behavioral evaluations, and per-seed analysis outputs behind every
   reported number are included under `artifacts/`. See
   [Reproduce the paper's figures and tables](#reproduce-the-papers-figures-and-tables).
2. **From scratch (about 330 GPU-hours on one 48 GB GPU).** Sections 1-4 re-run
   history formation, the continual stream, and relearning.

All commands are local (`uv run` / bash). Scripts always set `FACTTRACE_ROOT`
to this directory.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- One CUDA GPU (bfloat16). Production runs used a single 48 GB GPU
  (L40 / L40S) and ~128 GB host RAM per training job
- Hugging Face access to `Qwen/Qwen2.5-1.5B`

Sequential wall time is long: Phase A + dense Phase B is hours per seed;
confirmatory relearning is many independent cells (2-arm: 3 seeds × 2 runs ×
10 stream steps; 3-arm: 3 × 3 × 10). Cells can be run one after another on a
single GPU. The full project, including pilots, used about 330 GPU-hours, of
which about 250 were relearning runs.

```bash
./scripts/setup_env.sh
# optional logging
export WANDB_API_KEY=...
export FACTTRACE_WANDB=1
```

Sanity check (no GPU weights required beyond what tests import):

```bash
uv run pytest
```

## Reproduce the paper's figures and tables

```bash
uv run python scripts/paper/run_all.py
```

This reads only the released files under `artifacts/` and `data/`, and writes
to `artifacts/paper/`:

| Paper item | Output | Script |
|------------|--------|--------|
| Figure 1 (behavioral forgetting along the stream) | `fig1_behavioral_timeline.pdf` | `scripts/paper/fig1_timeline.py` |
| Figure 2 (relearning at *k* = 100) | `fig2_relearn_curves_k100.pdf` | `scripts/paper/fig2_relearn_curves.py` |
| Table 2 (primary Learned−Control savings) | `tables/table2_primary_savings_k100` | `scripts/paper/make_tables.py` |
| Table 3 (three-history contrasts) | `tables/table3_three_history_contrasts_k100` | same |
| Table 4 (baseline adjustment) | `tables/table4_baseline_adjustment_k100` | same |
| Table 5 (history-formation gates) | `tables/table5_history_formation_gates` | same |
| Table 6 (savings across stream steps) | `tables/table6_savings_across_stream_steps` | same |
| Table 7 (two-history fraction of naive) | `tables/table7_fraction_of_naive_two_history` | same |
| Table 8 (three-history fraction of naive) | `tables/table8_fraction_of_naive_three_history_k100` | same |
| In-text statistics (permutation test, pooled bootstrap, slope of *S* on Δ*L*) | `analyses.{json,md}` | `scripts/paper/analyses.py` |
| Supplementary plots (dose-response forest, common support) | `fig3_dose_response_forest.pdf`, `fig_a1_common_support.pdf` | `scripts/paper/fig3_dose_forest.py`, `scripts/paper/fig_a1_common_support.py` |

Each table is written as `.csv` and `.md`. Table 1 is notation only.

The per-seed analysis JSON that the tables read can itself be recomputed from
the released curves (CPU only):

```bash
./scripts/run_p5_relearn.sh --split confirmatory --seeds 0,1,2 --analyze-only
./scripts/run_p5_relearn.sh --config configs/experiment/phase1_3arm.yaml \
  --split confirmatory --seeds 0,1,2 --analyze-only
for s in 0 1 2; do
  uv run python scripts/analyze_fraction_of_naive.py \
    --config configs/experiment/phase1.yaml --seed "$s" --force
  uv run python scripts/analyze_fraction_of_naive.py \
    --config configs/experiment/phase1.yaml \
    --panel-config configs/experiment/phase1_3arm.yaml --seed "$s" \
    --contrasts learned:control,learned:wrong,wrong:control --force
done
```

## Released results (already in the repo)

| Path | Contents |
|------|----------|
| `artifacts/runs/phase1/seed_*/c*/relearn/dense_early_slow1e5/confirmatory/` | Two-history relearning curves, every stream step (`curves.parquet`) |
| `artifacts/runs/phase1_3arm/seed_*/c*/relearn/dense_early_slow1e5/confirmatory/` | Three-history relearning curves |
| `artifacts/runs/phase1/seed_*/c*/relearn/dense_early_slow1e5_fresh/fresh_pool/` | Fresh-pool relearning curves at *k* ∈ {50, 100, 150} |
| `artifacts/runs/*/seed_*/c*/eval_dense_early/` | Per-fact behavioral scores at every stream checkpoint |
| `artifacts/runs/*/seed_*/c*/phase_a*`, `phase_b_dense_early/phase_b_meta.json` | History-formation gate records and run metadata |
| `artifacts/analysis/p5/` | Per-seed savings, baseline-adjustment, and fraction-of-naive outputs |
| `artifacts/analysis/js_noise_floor_dense_early.{json,pdf}` | JS measurement floor |
| `data/manifests/`, `data/manifests_3arm/` | Checkpoint manifests, screening and exposure audits |

Model checkpoints are not included (each is a full 1.5B-parameter model);
Sections 1-2 regenerate them.

Not included: the development-partition relearning curves used to fix the 0.9
threshold, and the exploratory recipe search that preceded the locked protocol
(alternative stream learning rates and the wrong-answer overwrite stream). The
development curves can be regenerated with
`./scripts/run_p5_relearn.sh --split development --seeds 0,1,2`; without them,
`analyze_fraction_of_naive.py` uses the locked 0.9 threshold.

## Frozen inputs (already in the repo)

| Path | Role |
|------|------|
| `data/facts/facts.{parquet,jsonl}` | Screened 576-fact panel (6 relations × 8 answers × 12 entities) |
| `data/facts/splits.json` | Development 384 / confirmatory 192 |
| `data/facts/assignments/` | 2-arm complementary histories (seeds 0–2, c0/c1) |
| `data/facts/assignments_3arm/` | 3-arm Latin square (seeds 0–2, c0/c1/c2) |
| `data/templates/` | Disjoint Phase A / behavioral / relearn template banks |
| `data/facts/fresh_pool.{parquet,jsonl}` | 192 never-seen screened facts |
| `data/eval/general_validation.txt` | Stream-health perplexity passages |

Regenerating the panel (`generate_facts` → `screen_facts` →
`build_templates_and_splits` → `assign_histories`) is GPU-heavy and is **not**
bit-identical to this freeze. Use `--force` only if you intend a new draw.

```bash
# optional regeneration (not required to replicate the paper panel)
uv run python scripts/generate_facts.py --config configs/experiment/phase1.yaml --force
uv run python scripts/screen_facts.py --config configs/experiment/phase1.yaml --force
uv run python scripts/build_templates_and_splits.py --config configs/experiment/phase1.yaml --force
uv run python scripts/assign_histories.py --config configs/experiment/phase1.yaml --force
uv run python scripts/assign_histories.py --config configs/experiment/phase1_3arm.yaml --mode three_arm --force
uv run python scripts/generate_fresh_pool.py --force
```

## Locked protocol

| Setting | Value |
|---------|--------|
| Model | `Qwen/Qwen2.5-1.5B`, full fine-tune, bfloat16 |
| Seeds | 0, 1, 2 |
| Phase B | `dense_early` (400 updates, checkpoints every 25 steps) |
| Relearn | `configs/train/relearn.yaml`, tag `dense_early_slow1e5`, lr `1e-5` |
| Stream grid *k* | `{0,25,50,75,100,125,150,200,250,400}` |
| Relearn eval *u* | `{0,1,2,4,8,16,32,64}`; primary window `{1,2,4,8,16}` |
| Primary trough | *k* = 100 |
| Confirmatory *N* | 192 facts |

## 1. Two-arm confirmatory (primary Learned−Control)

For each seed, form matched Learned / Control histories, run the dense
continual stream, then relearn independently from each checkpoint.

```bash
for s in 0 1 2; do
  ./scripts/run_2arm_seed.sh --seed "$s"
done

./scripts/run_p5_relearn.sh --split confirmatory --seeds 0,1,2
```

`--prepare-phase-b` on the P5 runner trains missing `dense_early` checkpoints
if you skip `run_2arm_seed.sh`. Analysis JSON/PDFs land in
`artifacts/analysis/p5/`.

## 2. Three-arm confirmatory (Learned−Wrong / Wrong−Control)

Same stream and relearn tag; Latin-square assignment with a Wrong history.

```bash
for s in 0 1 2; do
  ./scripts/run_3arm_seed.sh --seed "$s"
done

./scripts/run_p5_relearn.sh \
  --config configs/experiment/phase1_3arm.yaml \
  --split confirmatory --seeds 0,1,2
```

Contrasts default to `learned:control,learned:wrong,wrong:control`.

## 3. Fresh-pool / fraction of naive

Relearn the never-seen pool from existing **two-arm** `dense_early`
checkpoints at *k* ∈ {50, 100, 150}, then convert savings to a fraction of
naive acquisition cost.

```bash
./scripts/run_fresh_relearn.sh --seeds 0,1,2
```

This also runs `analyze_fraction_of_naive.py` for 2-arm and (if 3-arm curves
exist) 3-arm contrasts.

## 4. JS noise floor

Re-eval dense_early checkpoints with per-template answer probabilities, then
estimate the JS measurement floor.

```bash
./scripts/run_js_per_template.sh --seeds 0,1,2
```

Output: `artifacts/analysis/js_noise_floor_dense_early.{json,pdf}`.

## Outputs

Re-running Sections 1-4 overwrites the released files in place.

| Path | Contents |
|------|----------|
| `artifacts/runs/phase1/seed_*/c*/` | 2-arm Phase A/B checkpoints (gitignored), eval, relearn curves |
| `artifacts/runs/phase1_3arm/seed_*/c*/` | 3-arm runs |
| `artifacts/analysis/p5/` | Savings, baseline-conditioned, fraction-of-naive JSON/PDF |
| `artifacts/hf_cache/` | Hugging Face model cache (`HF_HOME`, gitignored) |
| `data/manifests/` / `data/manifests_3arm/` | Checkpoint nominations written at eval time |

Re-run analysis only (after curves exist):

```bash
./scripts/run_p5_relearn.sh --split confirmatory --seeds 0,1,2 --analyze-only
./scripts/run_p5_relearn.sh --config configs/experiment/phase1_3arm.yaml \
  --split confirmatory --seeds 0,1,2 --analyze-only
```

## Layout

```
configs/        experiment, model, train, eval YAML
data/           frozen panel, assignments, templates, manifests
artifacts/      released curves, evaluations, and analysis outputs
scripts/        CLIs and local orchestrators
scripts/paper/  figure, table, and in-text statistics generators
src/            library (facts, train, eval, analysis)
tests/          pytest suite
```

## Scoring and relearning details

- Answers are multi-token (2-4 tokens). An answer's score is the sum of its
  token log-probabilities under teacher forcing; the eight relation-valid
  answers are renormalized and the target log-odds taken against the other
  seven (`src/eval/behavioral.py`).
- Every relearning run restores the checkpoint weights and starts a new AdamW
  optimizer with a constant learning rate (no warmup, no weight decay). Each
  update is one full-batch step on the fact's two relearning statements
  (`src/train/relearn.py`).
- Updates-to-threshold interpolates the 0.9 crossing linearly in
  log2(*u* + 1) between the bracketing budgets; facts that never reach the
  threshold are right-censored at *u* = 64 (`src/analysis/hierarchical.py`).

## Citation

Zhe Zhou. *Binding-Specific Relearning Savings for Test-Time Continual
Learning*. NeurIPS 2026 Workshop: Towards Test-Time Continual Learning Agents
(TTCL), 2026.

```bibtex
@inproceedings{zhou2026binding,
  title     = {Binding-Specific Relearning Savings for Test-Time Continual Learning},
  author    = {Zhou, Zhe},
  booktitle = {NeurIPS 2026 Workshop: Towards Test-Time Continual Learning Agents (TTCL)},
  year      = {2026}
}
```

Contact: Zhe Zhou (zhou1106@cs.washington.edu).

## License

Code is released under the MIT License (see `LICENSE`). The base model
`Qwen/Qwen2.5-1.5B` is distributed separately under Apache 2.0.
