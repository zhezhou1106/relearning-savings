"""Figure A1 — Common support for baseline adjustment (2-arm, per seed)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    SEEDS,
    apply_style,
    ensure_results_dir,
    fact_savings_at_trough,
    ols_with_ci,
    save_fig,
)


def _per_arm_baseline_lo(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return learned and control absolute LO at trough for confirmatory facts."""
    from common import (
        EXP_2ARM,
        K_TROUGH,
        PHASE_B_TAG,
        REPO_ROOT,
        confirmatory_ids,
        discover_runs,
    )

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from util.paths import complementary_dir, eval_dirname

    keep = confirmatory_ids()
    learned, control = {}, {}
    for comp in discover_runs(EXP_2ARM, seed):
        path = (
            complementary_dir(EXP_2ARM, seed, comp)
            / eval_dirname(PHASE_B_TAG)
            / f"behavioral_{K_TROUGH}.parquet"
        )
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = df[df["fact_id"].astype(str).isin(keep)]
        for r in df.itertuples(index=False):
            fid = str(r.fact_id)
            if r.history == "learned":
                learned[fid] = float(r.target_log_odds)
            elif r.history == "control":
                control[fid] = float(r.target_log_odds)
    common = sorted(set(learned) & set(control))
    return (
        np.array([learned[f] for f in common], dtype=float),
        np.array([control[f] for f in common], dtype=float),
    )


def plot_figure_a1(facts_by_seed: dict[int, pd.DataFrame]) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)

    for ax, seed in zip(axes, SEEDS):
        fact = facts_by_seed[seed]
        d = fact["d_i"].to_numpy(dtype=float)
        s = fact["s_i"].to_numpy(dtype=float)
        ax.scatter(d, s, s=10, alpha=0.35, color="#1f77b4", edgecolors="none")

        fit = ols_with_ci(d, s)
        # extend line to d=0
        x_min = float(np.nanmin(d))
        x_max = float(np.nanmax(d))
        xs = np.linspace(min(x_min, 0.0), max(x_max, 0.0), 100)
        ys = fit["intercept"] + fit["slope"] * xs
        ax.plot(xs, ys, color="0.15", lw=1.3)
        # CI ribbon
        X = np.column_stack([np.ones(len(d)), d])
        resid = s - (fit["intercept"] + fit["slope"] * d)
        dof = max(len(d) - 2, 1)
        s2 = float(np.sum(resid**2) / dof)
        xtx_inv = np.linalg.inv(X.T @ X)
        tcrit = fit.get("tcrit", 1.96)
        se_mean = np.sqrt(
            np.array([s2 * (np.array([1.0, xi]) @ xtx_inv @ np.array([1.0, xi])) for xi in xs])
        )
        ax.fill_between(xs, ys - tcrit * se_mean, ys + tcrit * se_mean, color="0.5", alpha=0.2)

        ax.axvline(0.0, color="0.4", ls="--", lw=0.9)
        ax.axhline(0.0, color="0.7", lw=0.6, ls=":")

        # Show whether d=0 lies inside the observed support of d_i (density + rug),
        # and annotate each history's absolute baseline log-odds range.
        lo_l, lo_c = _per_arm_baseline_lo(seed)
        ymin, ymax = ax.get_ylim()
        # temporary ylim for rugs
        ax.set_ylim(min(float(np.nanmin(s)) - 0.5, -1), max(float(np.nanmax(s)) + 0.5, 1))
        ymin, ymax = ax.get_ylim()
        y_l = ymin + 0.03 * (ymax - ymin)
        y_c = ymin + 0.08 * (ymax - ymin)
        # Map arm LO onto the d-axis? No — show rugs of each arm's contribution by
        # plotting (LO_L - mean(LO_C?)) ... Simplest faithful approach: rug of d_i,
        # and colored side annotations of [min,max] LO per arm.
        ax.plot(d, np.full_like(d, y_l), "|", color="#1f77b4", alpha=0.4, ms=7, mew=0.55)
        # Also rug learned/control absolute LO on the same axis only if we convert to
        # centered — instead draw LO rugs on a twin x at bottom using broken scale.
        # Use inset density of d_i and text for arm ranges:
        try:
            from scipy.stats import gaussian_kde

            if np.unique(d).size >= 3:
                kde = gaussian_kde(d)
                xs_d = np.linspace(float(d.min()), float(d.max()), 120)
                dens = kde(xs_d)
                dens = dens / dens.max() * 0.14 * (ymax - ymin)
                ax.fill_between(xs_d, ymax - dens, ymax, color="#1f77b4", alpha=0.22, linewidth=0)
        except Exception:
            pass

        # Per-arm support as vertical tick clusters at their mean LO? Review wants
        # whether d=0 falls inside each arm's observed range — that's about d_i per
        # arm which is a difference. Show range of d_i and of each arm's LO:
        ax.text(
            0.03,
            0.97,
            f"seed {seed}\n"
            f"intercept={fit['intercept']:.3f}\n"
            f"d∈[{d.min():.2f},{d.max():.2f}]\n"
            f"L LO∈[{lo_l.min():.2f},{lo_l.max():.2f}]\n"
            f"C LO∈[{lo_c.min():.2f},{lo_c.max():.2f}]",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.85", alpha=0.9),
        )
        # Colored rugs for arm absolute LO cannot share d-axis; draw them as
        # horizontal density strips using a secondary axis below — approximate by
        # mapping each arm's LO onto [d.min, d.max] via rank is wrong.
        # Draw LO rugs on twin axes at bottom of each panel:
        ax2 = ax.twiny()
        ax2.set_xlim(min(float(lo_l.min()), float(lo_c.min())) - 0.2, max(float(lo_l.max()), float(lo_c.max())) + 0.2)
        ax2.plot(lo_l, np.full_like(lo_l, 0.02), "|", color="#1f77b4", alpha=0.25, ms=5, mew=0.5, transform=ax2.get_xaxis_transform())
        ax2.plot(lo_c, np.full_like(lo_c, 0.06), "|", color="#ff7f0e", alpha=0.25, ms=5, mew=0.5, transform=ax2.get_xaxis_transform())
        ax2.set_xlabel("Arm absolute L (nats) — LEARNED blue / CONTROL orange rugs", fontsize=6.5)
        ax2.xaxis.set_label_position("top")
        ax2.tick_params(axis="x", labelsize=6)

        ax.set_xlabel(r"Baseline $d_i$ (behavioral-eval)")
        if seed == SEEDS[0]:
            ax.set_ylabel(r"Fact-level savings $s_i$")
        ax.set_title(f"Seed {seed}")

        # Mark if 0 outside support
        if d.min() > 0 or d.max() < 0:
            ax.set_facecolor("#fff8f0")

    fig.suptitle("Common support for baseline adjustment (2-arm, k=100)", fontsize=10, y=1.02)
    fig.tight_layout()
    return save_fig(fig, "fig_a1_common_support.pdf")


def main() -> Path:
    ensure_results_dir()
    facts_by_seed = {}
    for seed in SEEDS:
        fact = fact_savings_at_trough(seed)
        if fact.empty:
            raise SystemExit(f"No fact-level savings for seed {seed}")
        facts_by_seed[seed] = fact
        print(f"seed {seed}: n={len(fact)} d range=[{fact['d_i'].min():.3f},{fact['d_i'].max():.3f}]")
    out = plot_figure_a1(facts_by_seed)
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    main()
