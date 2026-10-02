#!/usr/bin/env python3
"""P5 dose-response analysis: relearning savings vs behavioral forgetting depth.

Supports two- and three-arm contrasts via --treatment/--reference, baseline-
conditioned intercept/slope regression, nonparametric residual contrast,
and updates-to-threshold savings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from analysis.figures import (
    plot_gain_vs_baseline_curves,
    plot_savings_vs_baseline,
    plot_savings_vs_forgetting,
    plot_savings_vs_stream_step,
)
from analysis.hierarchical import (
    PRIMARY_KS,
    choose_threshold_on_development,
    counterbalanced_bootstrap_ci,
    pooled_gain_residual_contrast,
    savings_baseline_regression,
    threshold_savings,
)
from util.config import load_yaml
from util.metrics import read_json, write_json
from util.paths import (
    complementary_dir,
    ensure_dir,
    eval_dirname,
    relearn_stream_dir,
    resolve_path,
    selected_checkpoints_path,
)


def _parse_steps(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _behavioral_row(manifest: dict, step: int) -> dict | None:
    for row in manifest.get("timeline") or []:
        if int(row["stream_steps"]) == step:
            return row
    return None


def _discover_runs(exp: str, seed: int) -> list[int]:
    seed_root = complementary_dir(exp, seed, 0).parent
    runs = []
    for p in sorted(seed_root.glob("c*")):
        if p.is_dir() and p.name.startswith("c") and p.name[1:].isdigit():
            runs.append(int(p.name[1:]))
    return runs or [0, 1]


def _curves_at_step(
    exp: str,
    seed: int,
    tag: str,
    step: int,
    split: str = "development",
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


def _all_curves(
    exp: str,
    seed: int,
    tag: str,
    steps: list[int],
    split: str,
    runs: list[int] | None = None,
) -> pd.DataFrame:
    parts = [_curves_at_step(exp, seed, tag, step, split, runs) for step in steps]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _behavioral_baseline_deltas(
    *,
    exp: str,
    seed: int,
    step: int,
    tag: str | None,
    fact_ids: set[str],
    treatment: str,
    reference: str,
    runs: list[int] | None = None,
) -> dict[str, float]:
    """d_i from the disjoint behavioral-eval template family (same checkpoint)."""
    by_arm: dict[str, dict[str, float]] = {treatment: {}, reference: {}}
    for comp in runs or _discover_runs(exp, seed):
        path = (
            complementary_dir(exp, seed, comp)
            / eval_dirname(tag)
            / f"behavioral_{step}.parquet"
        )
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = df[df["fact_id"].astype(str).isin(fact_ids)]
        for r in df.itertuples(index=False):
            hist = str(r.history)
            if hist in by_arm:
                by_arm[hist][str(r.fact_id)] = float(r.target_log_odds)
    common = sorted(set(by_arm[treatment]) & set(by_arm[reference]))
    return {
        fid: by_arm[treatment][fid] - by_arm[reference][fid] for fid in common
    }


def _strip_fact_level(reg: dict) -> dict:
    out = {k: v for k, v in reg.items() if k != "fact_level"}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/phase1.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tag", default="dense_early")
    parser.add_argument(
        "--phase-b-tag",
        default=None,
        help="Tag for selected_checkpoints manifest (default: --tag)",
    )
    parser.add_argument(
        "--relearn-tag",
        default=None,
        help="Tag for relearn curve directories (default: --tag)",
    )
    parser.add_argument("--split", choices=["development", "confirmatory"], default="development")
    parser.add_argument("--stream-steps", required=True)
    parser.add_argument("--treatment", default="learned")
    parser.add_argument("--reference", default="control")
    parser.add_argument(
        "--contrasts",
        default=None,
        help="Comma-separated treatment:reference pairs (overrides --treatment/--reference). "
        "Example: learned:control,learned:wrong,wrong:control",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dev-curves-tag",
        default=None,
        help="Optional relearn tag for development curves used to choose the "
        "updates-to-threshold (defaults to --relearn-tag on development split).",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    steps = _parse_steps(args.stream_steps)
    exp = str(cfg["experiment_name"])
    phase_b_tag = args.phase_b_tag or args.tag
    relearn_tag = args.relearn_tag or args.tag
    runs = _discover_runs(exp, args.seed)

    if args.contrasts:
        contrasts = []
        for pair in args.contrasts.split(","):
            treat, ref = pair.strip().split(":")
            contrasts.append((treat.strip(), ref.strip()))
    else:
        contrasts = [(args.treatment, args.reference)]

    manifest_path = selected_checkpoints_path(
        cfg["paths"]["manifests_dir"], args.seed, tag=phase_b_tag
    )
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    keep = {str(x) for x in read_json(resolve_path(cfg["paths"]["splits"]))[args.split]}

    out_dir = ensure_dir(resolve_path(cfg["paths"]["artifacts_dir"]) / "analysis" / "p5")
    contrast_suffix = (
        ""
        if contrasts == [("learned", "control")]
        else "_" + "_".join(f"{t}-vs-{r}" for t, r in contrasts)
    )
    out_json = (
        out_dir
        / f"savings_vs_forgetting_seed_{args.seed}_{relearn_tag}_{args.split}{contrast_suffix}.json"
    )
    out_fig = (
        out_dir
        / f"savings_vs_forgetting_seed_{args.seed}_{relearn_tag}_{args.split}{contrast_suffix}.pdf"
    )
    out_stream = (
        out_dir
        / f"savings_vs_stream_step_seed_{args.seed}_{relearn_tag}_{args.split}{contrast_suffix}.pdf"
    )
    if out_json.exists() and not args.force:
        print(f"Skip (exists): {out_json}")
        payload = read_json(out_json)
        df = pd.DataFrame(payload.get("checkpoints") or [])
        if not df.empty and "early_contrast" in df.columns:
            plot_savings_vs_stream_step({args.seed: df}, out_path=out_stream)
            print(f"Wrote {out_stream}")
        return

    # Development curves for threshold selection (may be same as confirmatory tag).
    threshold_info: dict | None = None
    if args.split == "confirmatory":
        # Prefer an existing development sweep; else fall back on confirmatory.
        dev_tag = args.dev_curves_tag or relearn_tag
        # Try loading development curves at the trough for threshold choice.
        trough_step = 100
        dev_curves = _curves_at_step(exp, args.seed, dev_tag, trough_step, "development", runs)
        if dev_curves.empty:
            # Fall back: use confirmatory curves at trough for a provisional threshold.
            dev_curves = _curves_at_step(
                exp, args.seed, relearn_tag, trough_step, "confirmatory", runs
            )
        if not dev_curves.empty:
            threshold_info = choose_threshold_on_development(dev_curves)
    else:
        trough_curves = _curves_at_step(exp, args.seed, relearn_tag, 100, args.split, runs)
        if not trough_curves.empty:
            threshold_info = choose_threshold_on_development(trough_curves)

    chosen_threshold = (
        float(threshold_info["threshold"]) if threshold_info else 0.50
    )

    # Pool all stream steps once for the nonparametric residual contrast.
    pooled = _all_curves(exp, args.seed, relearn_tag, steps, args.split, runs)
    if not pooled.empty and "fact_id" in pooled.columns:
        pooled = pooled[pooled["fact_id"].astype(str).isin(keep)]

    contrast_payloads: dict[str, dict] = {}
    primary_rows: list[dict] = []

    for treatment, reference in contrasts:
        key = f"{treatment}_vs_{reference}"
        residual = pooled_gain_residual_contrast(
            pooled if not pooled.empty else pd.DataFrame(),
            treatment=treatment,
            reference=reference,
            primary_ks=PRIMARY_KS,
            checkpoint_role="forgetting_sweep",
            stream_steps=100,
            n_boot=2000,
            seed=int(cfg.get("seed", 42)) + args.seed,
        )

        rows = []
        trough_reg_primary = None
        trough_fact_df = None
        for step in steps:
            beh = _behavioral_row(manifest, step)
            curves = _curves_at_step(exp, args.seed, relearn_tag, step, split=args.split, runs=runs)
            if not curves.empty and "fact_id" in curves.columns:
                curves = curves[curves["fact_id"].astype(str).isin(keep)]
            row: dict = {
                "stream_steps": step,
                "delta_log_odds": float(beh["delta_log_odds"]) if beh else float("nan"),
                "delta_log_odds_ci": beh.get("delta_log_odds_ci") if beh else None,
                "mean_js": float(beh.get("mean_js", float("nan"))) if beh else float("nan"),
                "learned_exact_match": float(beh.get("learned_exact_match", float("nan")))
                if beh
                else float("nan"),
                "control_exact_match": float(beh.get("control_exact_match", float("nan")))
                if beh
                else float("nan"),
                "n_relearn_curves": int(len(curves)),
                "treatment": treatment,
                "reference": reference,
            }
            if curves.empty:
                row.update(
                    {
                        "early_contrast": float("nan"),
                        "early_contrast_ci": None,
                        "primary_pass": False,
                        "n_facts": 0,
                        "warning": "no_relearn_curves",
                    }
                )
            else:
                curves = curves.copy()
                curves["checkpoint_role"] = "forgetting_sweep"
                boot = counterbalanced_bootstrap_ci(
                    curves,
                    n_boot=1000,
                    seed=int(cfg.get("seed", 42)) + args.seed + step,
                    treatment=treatment,
                    reference=reference,
                    checkpoint_role="forgetting_sweep",
                    primary_ks=PRIMARY_KS,
                )
                # Behavioral-eval d_i (primary) + relearn-k0 sensitivity.
                beh_deltas = _behavioral_baseline_deltas(
                    exp=exp,
                    seed=args.seed,
                    step=step,
                    tag=phase_b_tag,
                    fact_ids=keep,
                    treatment=treatment,
                    reference=reference,
                    runs=runs,
                )
                reg_beh = savings_baseline_regression(
                    curves,
                    treatment=treatment,
                    reference=reference,
                    checkpoint_role="forgetting_sweep",
                    primary_ks=PRIMARY_KS,
                    baseline_deltas=beh_deltas if beh_deltas else None,
                    d_source="behavioral_eval" if beh_deltas else "relearn_k0",
                    n_boot=2000,
                    seed=int(cfg.get("seed", 42)) + args.seed + step + 17,
                )
                reg_k0 = savings_baseline_regression(
                    curves,
                    treatment=treatment,
                    reference=reference,
                    checkpoint_role="forgetting_sweep",
                    primary_ks=PRIMARY_KS,
                    baseline_deltas=None,
                    d_source="relearn_k0",
                    n_boot=2000,
                    seed=int(cfg.get("seed", 42)) + args.seed + step + 31,
                )
                thr = threshold_savings(
                    curves,
                    threshold=chosen_threshold,
                    treatment=treatment,
                    reference=reference,
                    n_boot=2000,
                    seed=int(cfg.get("seed", 42)) + args.seed + step + 53,
                )
                step_residual = pooled_gain_residual_contrast(
                    pooled if not pooled.empty else pd.DataFrame(),
                    treatment=treatment,
                    reference=reference,
                    primary_ks=PRIMARY_KS,
                    checkpoint_role="forgetting_sweep",
                    stream_steps=step,
                    n_boot=1000,
                    seed=int(cfg.get("seed", 42)) + args.seed + step + 71,
                )
                row.update(
                    {
                        "early_contrast": boot["early_contrast"],
                        "early_contrast_ci": boot["early_contrast_ci"],
                        "integrated_gain": boot.get("integrated_gain"),
                        "primary_pass": boot.get("primary_pass"),
                        "n_facts": boot.get("n_facts"),
                        "savings_at_zero_baseline": reg_beh["intercept"],
                        "savings_at_zero_baseline_ci": reg_beh["intercept_ci"],
                        "baseline_slope": reg_beh["slope"],
                        "baseline_slope_ci": reg_beh["slope_ci"],
                        "baseline_regression_pass": reg_beh["primary_pass"],
                        "baseline_d_source": reg_beh["d_source"],
                        "savings_at_zero_baseline_relearn_k0": reg_k0["intercept"],
                        "savings_at_zero_baseline_relearn_k0_ci": reg_k0["intercept_ci"],
                        "baseline_slope_relearn_k0": reg_k0["slope"],
                        "residual_contrast": step_residual.get("residual_contrast"),
                        "residual_contrast_ci": step_residual.get("residual_contrast_ci"),
                        "residual_contrast_pass": step_residual.get("primary_pass"),
                        "updates_to_threshold": thr,
                        "threshold": chosen_threshold,
                    }
                )
                if step == 100 or (
                    beh
                    and abs(float(beh.get("delta_log_odds", 1e9)))
                    == min(
                        abs(float(r.get("delta_log_odds", 1e9)))
                        for r in (manifest.get("timeline") or [{"delta_log_odds": 1e9}])
                    )
                ):
                    trough_reg_primary = reg_beh
                    trough_fact_df = reg_beh.get("fact_level")
            rows.append(row)

        df = pd.DataFrame(rows)
        deepest = df.loc[df["delta_log_odds"].abs().idxmin()] if not df.empty else None
        contrast_payloads[key] = {
            "treatment": treatment,
            "reference": reference,
            "checkpoints": rows,
            "deepest_forgetting_step": int(deepest["stream_steps"])
            if deepest is not None
            else None,
            "deepest_forgetting_delta_log_odds": float(deepest["delta_log_odds"])
            if deepest is not None
            else None,
            "savings_at_deepest": float(deepest["early_contrast"])
            if deepest is not None and pd.notna(deepest.get("early_contrast"))
            else None,
            "savings_at_deepest_ci": deepest.get("early_contrast_ci")
            if deepest is not None
            else None,
            "savings_at_deepest_pass": bool(deepest.get("primary_pass"))
            if deepest is not None
            else False,
            "savings_at_zero_baseline_at_deepest": deepest.get("savings_at_zero_baseline")
            if deepest is not None
            else None,
            "savings_at_zero_baseline_at_deepest_ci": deepest.get(
                "savings_at_zero_baseline_ci"
            )
            if deepest is not None
            else None,
            "baseline_slope_at_deepest": deepest.get("baseline_slope")
            if deepest is not None
            else None,
            "residual_contrast": residual.get("residual_contrast"),
            "residual_contrast_ci": residual.get("residual_contrast_ci"),
            "residual_contrast_pass": residual.get("primary_pass"),
            "threshold": chosen_threshold,
            "threshold_selection": threshold_info,
        }
        if treatment == contrasts[0][0] and reference == contrasts[0][1]:
            primary_rows = rows
            # Figures for the primary (first) contrast.
            if trough_fact_df is not None and trough_reg_primary is not None:
                plot_savings_vs_baseline(
                    trough_fact_df,
                    intercept=float(trough_reg_primary["intercept"]),
                    slope=float(trough_reg_primary["slope"]),
                    out_path=out_dir
                    / f"savings_vs_baseline_seed_{args.seed}_{relearn_tag}_{args.split}_{key}.pdf",
                    title=f"s vs d @ trough ({key}, seed {args.seed})",
                )
            if not pooled.empty:
                plot_gain_vs_baseline_curves(
                    pooled,
                    out_path=out_dir
                    / f"gain_vs_baseline_seed_{args.seed}_{relearn_tag}_{args.split}.pdf",
                    primary_ks=PRIMARY_KS,
                )

    payload = {
        "seed": args.seed,
        "tag": relearn_tag,
        "phase_b_tag": phase_b_tag,
        "split": args.split,
        "primary_ks": list(PRIMARY_KS),
        "stream_steps": steps,
        "runs": runs,
        "contrasts": [list(c) for c in contrasts],
        "threshold": chosen_threshold,
        "threshold_selection": threshold_info,
        "checkpoints": primary_rows,
        "by_contrast": {
            k: {kk: vv for kk, vv in v.items() if kk != "checkpoints"}
            | {
                "checkpoints": [
                    {ck: cv for ck, cv in row.items() if ck != "fact_level"}
                    for row in v.get("checkpoints", [])
                ]
            }
            for k, v in contrast_payloads.items()
        },
    }
    # Convenience mirrors for the first contrast (back-compat with existing readers).
    first_key = f"{contrasts[0][0]}_vs_{contrasts[0][1]}"
    first = contrast_payloads[first_key]
    for field in (
        "deepest_forgetting_step",
        "deepest_forgetting_delta_log_odds",
        "savings_at_deepest",
        "savings_at_deepest_ci",
        "savings_at_deepest_pass",
        "savings_at_zero_baseline_at_deepest",
        "savings_at_zero_baseline_at_deepest_ci",
        "baseline_slope_at_deepest",
        "residual_contrast",
        "residual_contrast_ci",
        "residual_contrast_pass",
    ):
        payload[field] = first.get(field)

    write_json(out_json, payload)
    df = pd.DataFrame(primary_rows)
    if not df.empty and "early_contrast" in df.columns:
        plot_savings_vs_forgetting(df, out_path=out_fig)
        plot_savings_vs_stream_step({args.seed: df}, out_path=out_stream)
        print(f"Wrote {out_fig}")
        print(f"Wrote {out_stream}")
    print(f"Wrote {out_json}")
    if payload.get("deepest_forgetting_step") is not None:
        print(
            f"Deepest forgetting @ k={payload['deepest_forgetting_step']} "
            f"dLO={payload['deepest_forgetting_delta_log_odds']:.3f} "
            f"savings={payload['savings_at_deepest']} "
            f"intercept@d=0={payload.get('savings_at_zero_baseline_at_deepest')} "
            f"slope={payload.get('baseline_slope_at_deepest')}"
        )


if __name__ == "__main__":
    main()
