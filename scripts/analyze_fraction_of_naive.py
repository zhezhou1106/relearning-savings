#!/usr/bin/env python3
"""Fraction-of-naive trials-saved using fresh-pool relearn curves.

Fresh-pool facts are disjoint from the confirmatory panel, so the denominator
is the fresh-arm median updates-to-threshold (same threshold as P5).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from analysis.hierarchical import (
    choose_threshold_on_development,
    threshold_savings,
    updates_to_threshold,
)
from util.config import load_yaml
from util.metrics import read_json, write_json
from util.paths import (
    complementary_dir,
    ensure_dir,
    relearn_stream_dir,
    resolve_path,
)


def _discover_runs(exp: str, seed: int) -> list[int]:
    seed_root = complementary_dir(exp, seed, 0).parent
    runs = []
    for p in sorted(seed_root.glob("c*")):
        if p.is_dir() and p.name.startswith("c") and p.name[1:].isdigit():
            runs.append(int(p.name[1:]))
    return runs or [0, 1]


def _load_curves(
    exp: str,
    seed: int,
    tag: str,
    step: int,
    split: str,
    runs: list[int] | None = None,
) -> pd.DataFrame:
    parts = []
    for comp in runs or _discover_runs(exp, seed):
        path = relearn_stream_dir(exp, seed, comp, tag, step, split) / "curves.parquet"
        if path.exists():
            parts.append(pd.read_parquet(path))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _fresh_updates(df: pd.DataFrame, threshold: float) -> pd.Series:
    utt = updates_to_threshold(df, threshold=threshold)
    if utt.empty:
        return pd.Series(dtype=float)
    # One value per fresh fact (median across complements if duplicated).
    return utt.groupby("fact_id")["updates"].median()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/phase1.yaml")
    parser.add_argument(
        "--panel-config",
        default=None,
        help="Config for treatment/reference curves (default: --config)",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tag", default="dense_early_slow1e5")
    parser.add_argument("--fresh-tag", default="dense_early_slow1e5_fresh")
    parser.add_argument("--split", default="confirmatory")
    parser.add_argument("--fresh-split", default="fresh_pool")
    parser.add_argument("--stream-steps", default="50,100,150")
    parser.add_argument(
        "--contrasts",
        default="learned:control",
        help="Comma-separated treatment:reference pairs",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    panel_cfg = load_yaml(args.panel_config or args.config)
    panel_exp = str(panel_cfg["experiment_name"])
    fresh_exp = str(cfg["experiment_name"])  # fresh lives under phase1
    # Fresh arm was run on phase1 checkpoints; keep loading from phase1.
    fresh_exp = "phase1"

    steps = [int(x.strip()) for x in args.stream_steps.split(",") if x.strip()]
    contrasts = []
    for pair in args.contrasts.split(","):
        t, r = pair.strip().split(":")
        contrasts.append((t.strip(), r.strip()))

    keep = {
        str(x)
        for x in read_json(resolve_path(panel_cfg["paths"]["splits"]))[args.split]
    }
    runs = _discover_runs(panel_exp, args.seed)
    fresh_runs = _discover_runs(fresh_exp, args.seed)

    # Threshold from development panel curves at first available fresh step's
    # companion development relearn if present; else use confirmatory default 0.9
    # from locked P5 (documented choose_threshold_on_development result).
    chosen = 0.9
    thr_info: dict = {"threshold": chosen, "source": "locked_p5_default"}
    # Prefer recomputing on development if curves exist.
    for probe in (100, 50, 150, 0):
        dev = _load_curves(panel_exp, args.seed, args.tag, probe, "development", runs)
        if not dev.empty and {"learned", "control"}.issubset(set(dev["history"].unique())):
            thr_info = choose_threshold_on_development(
                dev[dev["fact_id"].astype(str).isin(
                    {
                        str(x)
                        for x in read_json(resolve_path(panel_cfg["paths"]["splits"]))[
                            "development"
                        ]
                    }
                )],
                treatment="learned",
                reference="control",
            )
            chosen = float(thr_info["threshold"])
            break

    out_dir = ensure_dir(resolve_path(panel_cfg["paths"]["artifacts_dir"]) / "analysis" / "p5")
    contrast_suffix = "_" + "_".join(f"{t}-vs-{r}" for t, r in contrasts)
    out_json = (
        out_dir
        / f"fraction_of_naive_seed_{args.seed}_{args.tag}_{args.split}{contrast_suffix}.json"
    )
    if out_json.exists() and not args.force:
        print(f"Skip (exists): {out_json}")
        return

    by_contrast: dict = {}
    for treatment, reference in contrasts:
        key = f"{treatment}_vs_{reference}"
        rows = []
        for step in steps:
            panel = _load_curves(panel_exp, args.seed, args.tag, step, args.split, runs)
            fresh = _load_curves(
                fresh_exp, args.seed, args.fresh_tag, step, args.fresh_split, fresh_runs
            )
            if panel.empty:
                rows.append(
                    {
                        "stream_steps": step,
                        "warning": "no_panel_curves",
                    }
                )
                continue
            panel = panel[panel["fact_id"].astype(str).isin(keep)].copy()
            if treatment not in set(panel["history"]) or reference not in set(
                panel["history"]
            ):
                rows.append(
                    {
                        "stream_steps": step,
                        "warning": "missing_histories",
                        "histories": sorted(panel["history"].unique().tolist()),
                    }
                )
                continue
            fresh_u = (
                _fresh_updates(fresh, chosen) if not fresh.empty else pd.Series(dtype=float)
            )
            # Pass values-only Series so disjoint fact_ids use median fallback.
            fresh_for_thr = (
                pd.Series(fresh_u.to_numpy(dtype=float)) if len(fresh_u) else None
            )
            thr = threshold_savings(
                panel,
                threshold=chosen,
                treatment=treatment,
                reference=reference,
                n_boot=2000,
                seed=int(panel_cfg.get("seed", 42)) + args.seed + step + 53,
                fresh_updates=fresh_for_thr,
            )
            if len(fresh_u):
                thr["naive_n_facts"] = int(len(fresh_u))
                thr["naive_median_updates"] = float(fresh_u.median())
                thr["naive_mean_updates"] = float(fresh_u.mean())
            thr["stream_steps"] = step
            thr["n_fresh_curves"] = int(len(fresh))
            thr["n_panel_curves"] = int(len(panel))
            rows.append(thr)
        by_contrast[key] = {"treatment": treatment, "reference": reference, "steps": rows}

    payload = {
        "seed": args.seed,
        "panel_experiment": panel_exp,
        "fresh_experiment": fresh_exp,
        "tag": args.tag,
        "fresh_tag": args.fresh_tag,
        "split": args.split,
        "stream_steps": steps,
        "threshold": chosen,
        "threshold_selection": thr_info,
        "contrasts": [list(c) for c in contrasts],
        "by_contrast": by_contrast,
    }
    write_json(out_json, payload)
    print(f"Wrote {out_json}")
    for key, blk in by_contrast.items():
        for row in blk["steps"]:
            if "updates_saved" not in row:
                print(f"  {key} k={row.get('stream_steps')}: {row.get('warning')}")
                continue
            frac = row.get("fraction_of_naive")
            print(
                f"  {key} k={row['stream_steps']}: "
                f"saved={row['updates_saved']:.3f} "
                f"naive_med={row.get('naive_median_updates', float('nan')):.3f} "
                f"frac={frac if frac is None else f'{frac:.3f}'} "
                f"frac_CI={row.get('fraction_of_naive_ci')} "
                f"pass={row.get('primary_pass')}"
            )


if __name__ == "__main__":
    main()
