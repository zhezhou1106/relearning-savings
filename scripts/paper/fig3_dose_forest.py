"""Figure 3 — Savings vs forgetting depth (2-arm) + 3-arm forest at trough."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CONTRAST_LABELS,
    CONTRAST_ORDER,
    K_TROUGH,
    SEEDS,
    apply_style,
    ensure_results_dir,
    load_savings_2arm,
    load_savings_3arm,
    ols_with_ci,
    save_fig,
)

SEED_MARKERS = {0: "o", 1: "s", 2: "^"}
SEED_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}


def collect_2arm_points() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        payload = load_savings_2arm(seed)
        ckpts = payload["by_contrast"]["learned_vs_control"]["checkpoints"]
        for c in ckpts:
            rows.append(
                {
                    "seed": seed,
                    "stream_steps": int(c["stream_steps"]),
                    "delta_log_odds": float(c["delta_log_odds"]),
                    "early_contrast": float(c["early_contrast"]),
                }
            )
    return pd.DataFrame(rows)


def collect_forest_rows() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        payload = load_savings_3arm(seed)
        for cname in CONTRAST_ORDER:
            ckpts = payload["by_contrast"][cname]["checkpoints"]
            hit = next(c for c in ckpts if int(c["stream_steps"]) == K_TROUGH)
            ci = hit["early_contrast_ci"]
            rows.append(
                {
                    "seed": seed,
                    "cname": cname,
                    "label": CONTRAST_LABELS[cname],
                    "S": float(hit["early_contrast"]),
                    "ci_lo": float(ci[0]),
                    "ci_hi": float(ci[1]),
                }
            )
    return pd.DataFrame(rows)


def plot_panel_a(points: pd.DataFrame, ax: plt.Axes | None = None) -> tuple[plt.Axes, dict]:
    """Savings vs forgetting depth (2-arm)."""
    standalone = ax is None
    if standalone:
        apply_style()
        _, ax = plt.subplots(figsize=(5.2, 4.2))

    main = points[points["stream_steps"] >= 50].copy()
    ceiling = points[points["stream_steps"].isin([0, 25])].copy()

    for seed in SEEDS:
        sub = main[main["seed"] == seed]
        ax.scatter(
            sub["delta_log_odds"],
            sub["early_contrast"],
            marker=SEED_MARKERS[seed],
            color=SEED_COLORS[seed],
            s=36,
            label=f"seed {seed}",
            zorder=3,
        )
        trough = sub[sub["stream_steps"] == K_TROUGH]
        if not trough.empty:
            ax.scatter(
                trough["delta_log_odds"],
                trough["early_contrast"],
                marker=SEED_MARKERS[seed],
                facecolors="none",
                edgecolors="k",
                s=90,
                linewidths=1.2,
                zorder=4,
            )

    x = main["delta_log_odds"].to_numpy()
    y = main["early_contrast"].to_numpy()
    fit = ols_with_ci(x, y)
    xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 80)
    ys = fit["intercept"] + fit["slope"] * xs
    ax.plot(xs, ys, color="0.2", lw=1.4, zorder=2)

    X = np.column_stack([np.ones(len(x)), x])
    resid = y - (fit["intercept"] + fit["slope"] * x)
    dof = max(len(x) - 2, 1)
    s2 = float(np.sum(resid**2) / dof)
    xtx_inv = np.linalg.inv(X.T @ X)
    se_mean = np.sqrt(
        np.array([s2 * (np.array([1.0, xi]) @ xtx_inv @ np.array([1.0, xi])) for xi in xs])
    )
    tcrit = fit.get("tcrit", 1.96)
    ax.fill_between(xs, ys - tcrit * se_mean, ys + tcrit * se_mean, color="0.5", alpha=0.2, zorder=1)

    ax.text(
        0.04,
        0.96,
        f"β = {fit['slope']:.3f} [{fit['slope_ci_lo']:.3f}, {fit['slope_ci_hi']:.3f}] nats/nat",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.9),
    )
    ax.axhline(0.0, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel("ΔL at checkpoint (nats)")
    ax.set_ylabel("Savings S (nats)")
    ax.set_title("(a) Savings vs forgetting depth (2-arm)")

    handles = [
        Line2D(
            [0],
            [0],
            marker=SEED_MARKERS[s],
            color="w",
            markerfacecolor=SEED_COLORS[s],
            markersize=7,
            label=f"seed {s}",
        )
        for s in SEEDS
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="k",
            markersize=8,
            label=f"k={K_TROUGH}",
        )
    )
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=7)

    if not ceiling.empty:
        inset = ax.inset_axes([0.55, 0.55, 0.42, 0.38])
        for seed in SEEDS:
            sub = ceiling[ceiling["seed"] == seed]
            inset.scatter(
                sub["delta_log_odds"],
                sub["early_contrast"],
                marker=SEED_MARKERS[seed],
                color=SEED_COLORS[seed],
                s=22,
            )
        inset.set_title("k∈{0,25} ceiling", fontsize=7)
        inset.tick_params(labelsize=6)
        inset.axhline(0.0, color="0.6", lw=0.6, ls="--")

    if standalone:
        ax.figure.tight_layout()
    return ax, fit


def plot_panel_b(forest: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Three-arm forest plot at trough."""
    standalone = ax is None
    if standalone:
        apply_style()
        _, ax = plt.subplots(figsize=(5.2, 4.2))

    y = 0.0
    yticks = []
    ylabels = []
    for cname in CONTRAST_ORDER:
        sub = forest[forest["cname"] == cname]
        mean_s = float(sub["S"].mean())
        mean_lo = float(sub["ci_lo"].mean())
        mean_hi = float(sub["ci_hi"].mean())
        label = CONTRAST_LABELS[cname]
        if cname == "wrong_vs_control":
            label = label + "\n(negative control)"
        ax.errorbar(
            mean_s,
            y,
            xerr=[[mean_s - mean_lo], [mean_hi - mean_s]],
            fmt="D",
            color="k",
            capsize=3.5,
            markersize=6,
            zorder=3,
        )
        yticks.append(y)
        ylabels.append(f"{label} pooled")
        y += 1.0
        for _, row in sub.sort_values("seed").iterrows():
            ax.errorbar(
                row["S"],
                y,
                xerr=[[row["S"] - row["ci_lo"]], [row["ci_hi"] - row["S"]]],
                fmt="o",
                color=SEED_COLORS[int(row["seed"])],
                capsize=2.5,
                markersize=4.5,
            )
            yticks.append(y)
            ylabels.append(f"  seed {int(row['seed'])}")
            y += 1.0
        y += 0.55

    ax.axvline(0.0, color="0.5", lw=1.0, ls="--")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"Savings S at k={K_TROUGH} (nats, 95% CI)")
    ax.set_title("(b) Three contrasts at the trough")
    ax.text(
        0.98,
        0.02,
        "Note: s(L−W)=s(L−C)−s(W−C)\nper fact (not independent)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="0.35",
    )
    if standalone:
        ax.figure.tight_layout()
    return ax


def plot_figure3(points: pd.DataFrame, forest: pd.DataFrame) -> tuple[Path, dict]:
    apply_style()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(9.6, 4.4), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )
    _, fit = plot_panel_a(points, ax=ax_a)
    plot_panel_b(forest, ax=ax_b)
    fig.tight_layout()
    return save_fig(fig, "fig3_dose_response_forest.pdf"), fit


def plot_figure3_panels(points: pd.DataFrame, forest: pd.DataFrame) -> tuple[Path, Path, dict]:
    apply_style()
    _, fit = plot_panel_a(points)
    out_a = save_fig(plt.gcf(), "fig3a_savings_vs_forgetting.png")

    apply_style()
    plot_panel_b(forest)
    out_b = save_fig(plt.gcf(), "fig3b_forest_trough.png")
    return out_a, out_b, fit


def main() -> tuple[Path, Path, Path]:
    ensure_results_dir()
    points = collect_2arm_points()
    forest = collect_forest_rows()
    out_pdf, fit = plot_figure3(points, forest)
    out_a, out_b, _ = plot_figure3_panels(points, forest)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_a}")
    print(f"Wrote {out_b}")
    print(
        f"Slope β={fit['slope']:.4f} "
        f"[{fit['slope_ci_lo']:.4f}, {fit['slope_ci_hi']:.4f}] n={fit['n']}"
    )
    return out_pdf, out_a, out_b


if __name__ == "__main__":
    main()
