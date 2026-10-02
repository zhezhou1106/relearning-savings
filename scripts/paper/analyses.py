"""Supplementary numerical analyses reported in the paper (numbers, not figures)."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CHANCE_LO,
    EXP_2ARM,
    EXP_3ARM,
    K_TROUGH,
    PHASE_B_TAG,
    PRIMARY_KS,
    RESULTS_DIR,
    SEEDS,
    behavioral_baseline_deltas,
    confirmatory_ids,
    discover_runs,
    ensure_results_dir,
    fact_level_savings,
    fact_savings_at_trough,
    load_curves_3arm_trough,
    load_manifest_2arm,
    load_savings_2arm,
    load_savings_3arm,
    ols_with_ci,
    timeline_on_grid,
    write_json,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from util.paths import complementary_dir, eval_dirname  # noqa: E402


def _slope_s_on_dl() -> dict[str, Any]:
    rows = []
    for seed in SEEDS:
        payload = load_savings_2arm(seed)
        for c in payload["by_contrast"]["learned_vs_control"]["checkpoints"]:
            if int(c["stream_steps"]) < 50:
                continue
            rows.append(
                {
                    "seed": seed,
                    "stream_steps": int(c["stream_steps"]),
                    "delta_log_odds": float(c["delta_log_odds"]),
                    "S": float(c["early_contrast"]),
                }
            )
    df = pd.DataFrame(rows)
    fit = ols_with_ci(df["delta_log_odds"].to_numpy(), df["S"].to_numpy())
    return {
        "description": "OLS of S on ΔL for 2-arm checkpoints with k≥50 (all seeds)",
        "n_points": fit["n"],
        "intercept": fit["intercept"],
        "slope": fit["slope"],
        "slope_se": fit["slope_se"],
        "slope_ci_95": [fit["slope_ci_lo"], fit["slope_ci_hi"]],
        "points": rows,
    }


def _delta_l_range() -> dict[str, Any]:
    vals = []
    for seed in SEEDS:
        for row in timeline_on_grid(load_manifest_2arm(seed)):
            k = int(row["stream_steps"])
            if 50 <= k <= 250:
                vals.append(
                    {
                        "seed": seed,
                        "stream_steps": k,
                        "delta_log_odds": float(row["delta_log_odds"]),
                    }
                )
    arr = np.array([v["delta_log_odds"] for v in vals], dtype=float)
    return {
        "description": "ΔL range spanned by k∈[50,250] across 2-arm seeds",
        "min": float(arr.min()),
        "max": float(arr.max()),
        "span": float(arr.max() - arr.min()),
        "values": vals,
    }


def _bootstrap_se_comparison() -> dict[str, Any]:
    """Compare bootstrap SE (CI half-width / 1.96) for L−C vs L−W at k=100."""
    per_seed = []
    for seed in SEEDS:
        payload = load_savings_3arm(seed)
        row: dict[str, Any] = {"seed": seed}
        for cname in ("learned_vs_control", "learned_vs_wrong"):
            hit = next(
                c
                for c in payload["by_contrast"][cname]["checkpoints"]
                if int(c["stream_steps"]) == K_TROUGH
            )
            lo, hi = hit["early_contrast_ci"]
            half = 0.5 * (float(hi) - float(lo))
            se = half / 1.96
            row[cname] = {
                "S": float(hit["early_contrast"]),
                "ci": [float(lo), float(hi)],
                "ci_halfwidth": half,
                "approx_se": se,
            }
        row["se_ratio_LC_over_LW"] = (
            row["learned_vs_control"]["approx_se"] / row["learned_vs_wrong"]["approx_se"]
            if row["learned_vs_wrong"]["approx_se"] > 0
            else float("nan")
        )
        per_seed.append(row)
    mean_se_lc = float(np.mean([r["learned_vs_control"]["approx_se"] for r in per_seed]))
    mean_se_lw = float(np.mean([r["learned_vs_wrong"]["approx_se"] for r in per_seed]))
    return {
        "description": "Approx bootstrap SE from 95% CI half-width/1.96 at k=100 (3-arm)",
        "per_seed": per_seed,
        "mean_se_learned_vs_control": mean_se_lc,
        "mean_se_learned_vs_wrong": mean_se_lw,
        "mean_se_ratio_LC_over_LW": mean_se_lc / mean_se_lw if mean_se_lw else float("nan"),
        "lw_tighter": mean_se_lw < mean_se_lc,
    }


def _fact_level_three_contrasts(seed: int) -> dict[str, pd.DataFrame]:
    keep = confirmatory_ids()
    curves = load_curves_3arm_trough(seed)
    curves = curves.copy()
    curves["checkpoint_role"] = "forgetting_sweep"
    out = {}
    for treatment, reference in (
        ("learned", "control"),
        ("learned", "wrong"),
        ("wrong", "control"),
    ):
        key = f"{treatment}_vs_{reference}"
        beh = behavioral_baseline_deltas(
            exp=EXP_3ARM,
            seed=seed,
            step=K_TROUGH,
            tag=PHASE_B_TAG,
            fact_ids=keep,
            treatment=treatment,
            reference=reference,
        )
        fact = fact_level_savings(
            curves,
            treatment=treatment,
            reference=reference,
            checkpoint_role="forgetting_sweep",
            primary_ks=PRIMARY_KS,
            baseline_deltas=beh,
        )
        out[key] = fact.set_index("fact_id")
    return out


def _linear_dependence_residual() -> dict[str, Any]:
    per_seed = []
    global_max = 0.0
    for seed in SEEDS:
        parts = _fact_level_three_contrasts(seed)
        common = (
            set(parts["learned_vs_control"].index)
            & set(parts["learned_vs_wrong"].index)
            & set(parts["wrong_vs_control"].index)
        )
        residuals = []
        for fid in common:
            s_lc = float(parts["learned_vs_control"].loc[fid, "s_i"])
            s_lw = float(parts["learned_vs_wrong"].loc[fid, "s_i"])
            s_wc = float(parts["wrong_vs_control"].loc[fid, "s_i"])
            resid = s_lw - (s_lc - s_wc)
            residuals.append(resid)
        arr = np.asarray(residuals, dtype=float)
        max_abs = float(np.max(np.abs(arr))) if arr.size else float("nan")
        global_max = max(global_max, max_abs if np.isfinite(max_abs) else 0.0)
        per_seed.append(
            {
                "seed": seed,
                "n_facts": int(arr.size),
                "max_abs_residual": max_abs,
                "mean_abs_residual": float(np.mean(np.abs(arr))) if arr.size else float("nan"),
                "rmse": float(np.sqrt(np.mean(arr**2))) if arr.size else float("nan"),
            }
        )
    return {
        "description": "Verify s_i(L−W) − [s_i(L−C) − s_i(W−C)] = 0 fact-by-fact",
        "per_seed": per_seed,
        "global_max_abs_residual": global_max,
    }


def _early_gains_by_fact_history(curves: pd.DataFrame) -> pd.DataFrame:
    """Per-fact mean early-window gain for each history.

    gain_k = target_log_odds_k - baseline_log_odds (equiv. lo_k - lo_0).
    Mean over PRIMARY_KS. Then S = mean_i (g_i^L - g_i^C).
    """
    sub = curves[curves["k"].isin(PRIMARY_KS)].copy()
    if "gain" not in sub.columns:
        sub["gain"] = sub["target_log_odds"] - sub["baseline_log_odds"]
    g = (
        sub.groupby(["fact_id", "history"], as_index=False)["gain"]
        .mean()
        .rename(columns={"gain": "early_gain"})
    )
    wide = g.pivot(index="fact_id", columns="history", values="early_gain")
    for h in ("learned", "control", "wrong"):
        if h not in wide.columns:
            wide[h] = np.nan
    return wide.dropna(subset=["learned", "control", "wrong"])


def _within_fact_permutation(
    *, n_perm: int = 1999, rng_seed: int = 0
) -> dict[str, Any]:
    """Monte Carlo permutation test of L−C savings, permuting history labels within fact.

    Vectorized: precompute early gains per (fact, history), then for each perm
    randomly reassign the three gains to L/C/W within each fact.
    """
    rng = np.random.default_rng(rng_seed)
    per_seed = []
    p_values = []
    # All 6 permutations of columns [L, C, W]
    all_perms = list(itertools.permutations([0, 1, 2]))

    for seed in SEEDS:
        curves = load_curves_3arm_trough(seed)
        wide = _early_gains_by_fact_history(curves)
        mat = wide[["learned", "control", "wrong"]].to_numpy(dtype=float)  # (n, 3)
        n = mat.shape[0]
        obs = float((mat[:, 0] - mat[:, 1]).mean())

        # Vectorized: for each (perm, fact) pick a column permutation of gains
        perms_arr = np.asarray(all_perms, dtype=int)  # (6, 3)
        perm_idx = rng.integers(0, len(all_perms), size=(n_perm, n))
        selected = perms_arr[perm_idx]  # (n_perm, n, 3)
        # out[i, j, c] = mat[j, selected[i, j, c]]
        expanded = np.broadcast_to(mat[None, :, :], (n_perm, n, 3))
        out = np.take_along_axis(expanded, selected, axis=2)
        null = (out[:, :, 0] - out[:, :, 1]).mean(axis=1)

        p = (1.0 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1.0)
        p_values.append(float(p))
        per_seed.append(
            {
                "seed": seed,
                "observed_S": float(obs),
                "null_mean": float(null.mean()),
                "null_sd": float(null.std(ddof=1)),
                "p_two_sided": float(p),
                "n_perm": n_perm,
                "n_facts": int(n),
            }
        )
    chi2 = -2.0 * np.sum(np.log(np.clip(p_values, 1e-300, 1.0)))
    try:
        from scipy import stats

        combined_p = float(stats.chi2.sf(chi2, df=2 * len(p_values)))
    except Exception:
        combined_p = float("nan")
    return {
        "description": (
            "Within-fact permutation of {L,C,W} early-gain labels; S = mean(g^L−g^C); "
            "Monte Carlo two-sided p; Fisher combination across seeds"
        ),
        "per_seed": per_seed,
        "fisher_chi2": float(chi2),
        "combined_p": combined_p,
    }


def _pooled_cluster_bootstrap(
    *, n_boot: int = 2000, rng_seed: int = 0, alpha: float = 0.05
) -> dict[str, Any]:
    """Cluster bootstrap over seeds and facts for headline L−C S at k=100 (2-arm)."""
    by_seed: dict[int, pd.DataFrame] = {}
    for seed in SEEDS:
        fact = fact_savings_at_trough(seed, exp=EXP_2ARM)
        by_seed[seed] = fact.set_index("fact_id")
    # point: mean of seed means
    seed_means = {s: float(df["s_i"].mean()) for s, df in by_seed.items()}
    point = float(np.mean(list(seed_means.values())))

    rng = np.random.default_rng(rng_seed)
    seeds = list(SEEDS)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_ests = []
        for s in sampled_seeds:
            df = by_seed[int(s)]
            fids = df.index.to_numpy()
            sample = rng.choice(fids, size=len(fids), replace=True)
            seed_ests.append(float(df.loc[sample, "s_i"].mean()))
        boots[i] = float(np.mean(seed_ests))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {
        "description": "Pooled cluster-bootstrap of L−C fact-level s_i at k=100 (2-arm)",
        "point_estimate": point,
        "per_seed_means": seed_means,
        "ci_95": [float(lo), float(hi)],
        "n_boot": n_boot,
        "se": float(boots.std(ddof=1)),
    }


def _baseline_l_distribution() -> dict[str, Any]:
    """Proxy for base-model L: CONTROL-arm target LO at stream 0 (no target training).

    Screening did not persist per-fact base-model log-odds; control at k=0 is the
    closest on-disk measure of residual mass on retained panel facts.
    """
    keep = confirmatory_ids()
    # also include development for full panel of 576
    splits = __import__("json").loads(
        (Path(__file__).resolve().parents[2] / "data" / "facts" / "splits.json").read_text()
    )
    all_panel = {str(x) for x in splits["development"]} | {str(x) for x in splits["confirmatory"]}

    vals_control = []
    vals_all_hist_control_dev = []
    for seed in SEEDS:
        for comp in discover_runs(EXP_2ARM, seed):
            path = (
                complementary_dir(EXP_2ARM, seed, comp)
                / eval_dirname(PHASE_B_TAG)
                / "behavioral_0.parquet"
            )
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            ctrl = df[(df["history"] == "control") & (df["fact_id"].astype(str).isin(all_panel))]
            vals_control.extend(ctrl["target_log_odds"].astype(float).tolist())
            # confirmatory only subset
            ctrl_c = df[(df["history"] == "control") & (df["fact_id"].astype(str).isin(keep))]
            vals_all_hist_control_dev.extend(ctrl_c["target_log_odds"].astype(float).tolist())

    arr = np.asarray(vals_control, dtype=float)
    arr_c = np.asarray(vals_all_hist_control_dev, dtype=float)
    return {
        "description": (
            "CONTROL-arm target log-odds at stream_steps=0 (2-arm behavioral eval). "
            "Base-model screening L not persisted; this is the no-target-training proxy. "
            f"Chance ≈ {CHANCE_LO:.4f} nats."
        ),
        "chance_log_odds": CHANCE_LO,
        "panel_all_seeds_pooled": {
            "n": int(arr.size),
            "mean": float(arr.mean()) if arr.size else float("nan"),
            "sd": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
        },
        "confirmatory_only_pooled": {
            "n": int(arr_c.size),
            "mean": float(arr_c.mean()) if arr_c.size else float("nan"),
            "sd": float(arr_c.std(ddof=1)) if arr_c.size > 1 else float("nan"),
        },
        "note": "Values pooled across seeds×complementary runs (not independent).",
    }


def run_all_analyses() -> dict[str, Any]:
    def _log(msg: str) -> None:
        print(msg, flush=True)

    _log("Computing slope S ~ ΔL...")
    slope = _slope_s_on_dl()
    _log("Computing ΔL range...")
    dl_range = _delta_l_range()
    _log("Computing bootstrap SE comparison...")
    se_cmp = _bootstrap_se_comparison()
    _log("Computing linear-dependence residual...")
    lindep = _linear_dependence_residual()
    _log("Computing baseline L distribution...")
    base_l = _baseline_l_distribution()
    _log("Computing pooled cluster-bootstrap...")
    pooled = _pooled_cluster_bootstrap()
    _log("Computing within-fact permutation test...")
    perm = _within_fact_permutation(n_perm=1999)

    return {
        "within_fact_permutation": perm,
        "pooled_cluster_bootstrap": pooled,
        "slope_S_on_deltaL": slope,
        "linear_dependence_residual": lindep,
        "bootstrap_se_LC_vs_LW": se_cmp,
        "baseline_L_distribution": base_l,
        "deltaL_range_k50_250": dl_range,
    }


def _to_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Supplementary numerical analyses", ""]
    # Permutation
    p = payload["within_fact_permutation"]
    lines.append("## Within-fact permutation test (3-arm L−C)")
    lines.append(p["description"])
    lines.append("")
    for row in p["per_seed"]:
        lines.append(
            f"- Seed {row['seed']}: S={row['observed_S']:.4f}, "
            f"p={row['p_two_sided']:.4f} ({row['n_perm']} perms)"
        )
    lines.append(f"- Combined (Fisher) p={p['combined_p']:.4g}")
    lines.append("")

    b = payload["pooled_cluster_bootstrap"]
    lines.append("## Pooled cluster-bootstrap headline S (2-arm L−C @ k=100)")
    lines.append(
        f"- Point={b['point_estimate']:.4f}, "
        f"95% CI=[{b['ci_95'][0]:.4f}, {b['ci_95'][1]:.4f}], "
        f"SE≈{b['se']:.4f}"
    )
    lines.append(f"- Per-seed means: {b['per_seed_means']}")
    lines.append("")

    s = payload["slope_S_on_deltaL"]
    lines.append("## Slope of S on ΔL (k≥50, 2-arm)")
    lines.append(
        f"- β={s['slope']:.4f}, SE={s['slope_se']:.4f}, "
        f"95% CI=[{s['slope_ci_95'][0]:.4f}, {s['slope_ci_95'][1]:.4f}], n={s['n_points']}"
    )
    lines.append("")

    r = payload["linear_dependence_residual"]
    lines.append("## Linear-dependence residual check")
    lines.append(r["description"])
    for row in r["per_seed"]:
        lines.append(
            f"- Seed {row['seed']}: max|resid|={row['max_abs_residual']:.3e}, "
            f"RMSE={row['rmse']:.3e}"
        )
    lines.append(f"- Global max|resid|={r['global_max_abs_residual']:.3e}")
    lines.append("")

    se = payload["bootstrap_se_LC_vs_LW"]
    lines.append("## Bootstrap SE comparison L−C vs L−W")
    lines.append(
        f"- Mean SE L−C={se['mean_se_learned_vs_control']:.4f}, "
        f"L−W={se['mean_se_learned_vs_wrong']:.4f}, "
        f"ratio={se['mean_se_ratio_LC_over_LW']:.3f}, "
        f"L−W tighter={se['lw_tighter']}"
    )
    lines.append("")

    bl = payload["baseline_L_distribution"]
    lines.append("## Baseline L distribution (proxy)")
    lines.append(bl["description"])
    pan = bl["panel_all_seeds_pooled"]
    lines.append(
        f"- Panel control@k=0: mean={pan['mean']:.3f} ± {pan['sd']:.3f} "
        f"(n={pan['n']}); chance={bl['chance_log_odds']:.3f}"
    )
    lines.append("")

    d = payload["deltaL_range_k50_250"]
    lines.append("## ΔL range for k=50–250 (2-arm)")
    lines.append(f"- [{d['min']:.4f}, {d['max']:.4f}] (span={d['span']:.4f})")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> tuple[Path, Path]:
    ensure_results_dir()
    payload = run_all_analyses()
    json_path = write_json(RESULTS_DIR / "analyses.json", payload)
    md_path = RESULTS_DIR / "analyses.md"
    md_path.write_text(_to_markdown(payload))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return json_path, md_path


if __name__ == "__main__":
    main()
