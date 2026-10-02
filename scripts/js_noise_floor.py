#!/usr/bin/env python3
"""JS noise floor: same-history cross-seed JS vs cross-arm JS and margin lines.

Upper bound: JS between independently-seeded runs of the same history at the
same checkpoint (uses persisted answer_probs — no GPU).

Lower bound (optional): between-template JS within a single model, requiring
``answer_probs_per_template`` from ``eval_retention.py --per-template``.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from analysis.figures import plot_js_noise_floor
from eval.behavioral import js_distance
from util.config import load_yaml
from util.metrics import read_json, write_json
from util.paths import (
    assignment_path,
    complementary_dir,
    ensure_dir,
    eval_dirname,
    resolve_path,
    selected_checkpoints_path,
)


def _load_probs(
    *,
    exp: str,
    seed: int,
    step: int,
    tag: str | None,
    assignments_dir: str,
    facts_parquet: str,
) -> dict[str, dict[str, np.ndarray]]:
    """fact_id -> history -> target-aligned 8-way probs (target at index 0)."""
    facts_df = pd.read_parquet(resolve_path(facts_parquet))
    answer_ids = {str(r.fact_id): int(r.answer_id) for r in facts_df.itertuples()}
    out: dict[str, dict[str, np.ndarray]] = {}
    for comp in (0, 1):
        path = (
            complementary_dir(exp, seed, comp)
            / eval_dirname(tag)
            / f"behavioral_{step}.parquet"
        )
        if not path.exists():
            continue
        asg = read_json(assignment_path(assignments_dir, seed, comp))
        df = pd.read_parquet(path)
        for r in df.itertuples(index=False):
            fid = str(r.fact_id)
            hist = str(asg[fid]["history"]) if fid in asg else str(r.history)
            probs = np.asarray(list(r.answer_probs), dtype=np.float64)
            aid = answer_ids.get(fid, 0)
            aligned = np.roll(probs, -aid)
            out.setdefault(fid, {})[hist] = aligned
    return out


def _bootstrap_mean_ci(
    values: np.ndarray, *, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, list[float]]:
    if values.size == 0:
        return 0.0, [0.0, 0.0]
    rng = np.random.default_rng(seed)
    point = float(values.mean())
    draws = np.array(
        [rng.choice(values, size=values.size, replace=True).mean() for _ in range(n_boot)]
    )
    lo, hi = alpha / 2, 1 - alpha / 2
    return point, [float(np.quantile(draws, lo)), float(np.quantile(draws, hi))]


def _measurement_floor(
    *,
    exp: str,
    seeds: list[int],
    steps: list[int],
    tag: str | None,
) -> dict:
    """Between-template JS within a single model (measurement noise only)."""
    vals: list[float] = []
    n_files = 0
    for seed in seeds:
        for comp in (0, 1):
            for step in steps:
                path = (
                    complementary_dir(exp, seed, comp)
                    / eval_dirname(tag)
                    / f"behavioral_{step}.parquet"
                )
                if not path.exists():
                    continue
                df = pd.read_parquet(path)
                if "answer_probs_per_template" not in df.columns:
                    continue
                n_files += 1
                for r in df.itertuples(index=False):
                    per = list(r.answer_probs_per_template)
                    if len(per) < 2:
                        continue
                    vals.append(js_distance(per[0], per[1]))
    arr = np.asarray(vals, dtype=float)
    point, ci = _bootstrap_mean_ci(arr, seed=7)
    return {
        "mean_js": point,
        "mean_js_ci": ci,
        "n_pairs": int(arr.size),
        "n_files": n_files,
        "note": "JS between the two behavioral templates of a single model",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/phase1.yaml")
    parser.add_argument("--tag", default="dense_early")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--steps",
        default="0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    exp = str(cfg["experiment_name"])
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    steps = [int(x) for x in args.steps.split(",") if x.strip()]
    out_dir = ensure_dir(resolve_path(cfg["paths"]["artifacts_dir"]) / "analysis")
    out_json = out_dir / f"js_noise_floor_{args.tag}.json"
    out_fig = out_dir / f"js_noise_floor_{args.tag}.pdf"
    if out_json.exists() and not args.force:
        print(f"Skip (exists): {out_json}")
        return

    null_rows = []
    for step in steps:
        per_hist_vals: dict[str, list[float]] = {"learned": [], "control": []}
        for s_a, s_b in combinations(seeds, 2):
            probs_a = _load_probs(
                exp=exp,
                seed=s_a,
                step=step,
                tag=args.tag,
                assignments_dir=cfg["paths"]["assignments_dir"],
                facts_parquet=cfg["paths"]["facts_parquet"],
            )
            probs_b = _load_probs(
                exp=exp,
                seed=s_b,
                step=step,
                tag=args.tag,
                assignments_dir=cfg["paths"]["assignments_dir"],
                facts_parquet=cfg["paths"]["facts_parquet"],
            )
            common = sorted(set(probs_a) & set(probs_b))
            for hist in ("learned", "control"):
                for fid in common:
                    if hist in probs_a[fid] and hist in probs_b[fid]:
                        per_hist_vals[hist].append(
                            js_distance(probs_a[fid][hist], probs_b[fid][hist])
                        )
        all_vals = np.asarray(
            per_hist_vals["learned"] + per_hist_vals["control"], dtype=float
        )
        point, ci = _bootstrap_mean_ci(all_vals, seed=42 + step)
        null_rows.append(
            {
                "stream_steps": step,
                "mean_js": point,
                "mean_js_ci": ci,
                "n_pairs": int(all_vals.size),
                "learned_mean_js": float(np.mean(per_hist_vals["learned"]))
                if per_hist_vals["learned"]
                else float("nan"),
                "control_mean_js": float(np.mean(per_hist_vals["control"]))
                if per_hist_vals["control"]
                else float("nan"),
            }
        )

    # Cross-arm timeline from seed-0 dense_early manifest (representative).
    cross_arm_rows = []
    for seed in seeds:
        man_path = selected_checkpoints_path(
            cfg["paths"]["manifests_dir"], seed, tag=args.tag
        )
        if not man_path.exists():
            continue
        man = read_json(man_path)
        for row in man.get("timeline") or []:
            cross_arm_rows.append(
                {
                    "seed": seed,
                    "stream_steps": int(row["stream_steps"]),
                    "mean_js": float(row.get("mean_js", float("nan"))),
                }
            )
    cross_arm = pd.DataFrame(cross_arm_rows)
    if not cross_arm.empty:
        cross_arm = (
            cross_arm.groupby("stream_steps", as_index=False)["mean_js"].mean()
        )

    # Measurement-noise lower bound (only where per-template columns exist).
    common_steps = [0, 25, 50, 75, 100, 125, 150, 200, 250, 400]
    measurement = _measurement_floor(
        exp=exp, seeds=seeds, steps=common_steps, tag=args.tag
    )

    null_df = pd.DataFrame(null_rows)
    # Drop steps with no pairs (missing evals).
    null_df = null_df[null_df["n_pairs"] > 0].reset_index(drop=True)

    payload = {
        "tag": args.tag,
        "seeds": seeds,
        "steps": steps,
        "null_timeline": null_rows,
        "null_min_mean_js": float(null_df["mean_js"].min()) if not null_df.empty else None,
        "null_at_js_trough": None,
        "pilot_margin": 0.03,
        "confirmatory_margin": 0.02,
        "measurement_floor": measurement,
        "caveat": (
            "Cross-seed same-history JS is an UPPER bound on pure seed noise: "
            "Phase A stop steps differ across seeds and history assignment is "
            "reshuffled. The between-template measurement floor is a LOWER bound."
        ),
    }
    if not null_df.empty:
        # Closest null to the observed JS trough around k=150.
        near = null_df.iloc[(null_df["stream_steps"] - 150).abs().argsort()[:1]]
        if len(near):
            payload["null_at_js_trough"] = {
                "stream_steps": int(near.iloc[0]["stream_steps"]),
                "mean_js": float(near.iloc[0]["mean_js"]),
                "mean_js_ci": near.iloc[0]["mean_js_ci"],
            }

    write_json(out_json, payload)
    plot_js_noise_floor(
        null_timeline=null_df,
        cross_arm_timeline=cross_arm if not cross_arm.empty else None,
        out_path=out_fig,
        pilot_margin=0.03,
        confirmatory_margin=0.02,
        measurement_floor=measurement["mean_js"]
        if measurement.get("n_pairs", 0) > 0
        else None,
    )
    print(f"Wrote {out_json}")
    print(f"Wrote {out_fig}")
    if payload["null_min_mean_js"] is not None:
        print(
            f"Null min mean JS={payload['null_min_mean_js']:.4f}; "
            f"pilot margin=0.03; confirmatory=0.02; "
            f"measurement_floor={measurement.get('mean_js')}"
        )


if __name__ == "__main__":
    main()
