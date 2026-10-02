"""Central retention vs relearning figure (phase-1 study)."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util.paths import ensure_dir


def _ci_yerr(sub: pd.DataFrame, y_col: str, ci_col: str) -> list[list[float]] | None:
    if ci_col not in sub.columns:
        return None
    lowers, uppers = [], []
    any_width = False
    for _, row in sub.iterrows():
        ci = row.get(ci_col)
        y = float(row[y_col])
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            lo = y - float(ci[0])
            hi = float(ci[1]) - y
            any_width = any_width or lo > 0 or hi > 0
            lowers.append(max(lo, 0.0))
            uppers.append(max(hi, 0.0))
        else:
            lowers.append(0.0)
            uppers.append(0.0)
    if not any_width:
        return None
    return [lowers, uppers]


def plot_central_figure(
    *,
    retention_df: pd.DataFrame,
    relearn_df: pd.DataFrame | None = None,
    out_path: str | Path,
    equivalence_margin: float = 0.15,
    confirmed_steps: int | None = None,
) -> Path:
    """Panel A: behavioral retention over stream. Panel B: relearning curves."""
    out = Path(out_path)
    ensure_dir(out.parent)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    ax.axhline(0.0, color="0.5", lw=1, ls="--")
    ax.axhspan(-equivalence_margin, equivalence_margin, color="0.85", alpha=0.5, label="ΔL band")
    if not retention_df.empty and "delta_behavior" in retention_df.columns:
        ax.plot(
            retention_df["stream_steps"],
            retention_df["delta_behavior"],
            marker="o",
            label="Δ log-odds",
        )
    if confirmed_steps is not None:
        ax.axvline(confirmed_steps, color="C3", ls=":", label=f"confirmed @{confirmed_steps}")
    ax.set_xlabel("Phase-B stream updates")
    ax.set_ylabel("Learned − control")
    ax.set_title("A. Behavioral retention")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    if relearn_df is not None and not relearn_df.empty:
        for hist, style in (("learned", "-"), ("control", "--")):
            sub = relearn_df[relearn_df["history"] == hist]
            if sub.empty:
                continue
            means = sub.groupby("k")["target_log_odds"].mean()
            ax.plot(means.index, means.values, ls=style, marker="o", label=hist)
    ax.set_xlabel("Relearning updates k")
    ax.set_ylabel("Target log-odds")
    ax.set_title("B. Relearning at confirmed checkpoint")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_savings_vs_forgetting(
    df: pd.DataFrame,
    *,
    out_path: str | Path,
    x_col: str = "delta_log_odds",
    y_col: str = "early_contrast",
    step_col: str = "stream_steps",
) -> Path:
    """Savings (baseline-adjusted relearning contrast) vs behavioral forgetting."""
    out = Path(out_path)
    ensure_dir(out.parent)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    sub = df.dropna(subset=[x_col, y_col]).copy()
    if not sub.empty:
        yerr = None
        if "early_contrast_ci" in sub.columns:
            lowers, uppers = [], []
            for _, row in sub.iterrows():
                ci = row.get("early_contrast_ci")
                if isinstance(ci, (list, tuple)) and len(ci) == 2:
                    y = float(row[y_col])
                    lowers.append(y - float(ci[0]))
                    uppers.append(float(ci[1]) - y)
                else:
                    lowers.append(0.0)
                    uppers.append(0.0)
            if any(lo > 0 or hi > 0 for lo, hi in zip(lowers, uppers)):
                yerr = [lowers, uppers]
        ax.errorbar(
            sub[x_col],
            sub[y_col],
            yerr=yerr,
            fmt="o",
            capsize=3,
            label="checkpoint",
        )
        for _, row in sub.iterrows():
            ax.annotate(
                str(int(row[step_col])),
                (row[x_col], row[y_col]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )
        ax.axhline(0.0, color="0.5", lw=1, ls="--")
        idx = sub[x_col].abs().idxmin()
        ax.scatter(
            [sub.loc[idx, x_col]],
            [sub.loc[idx, y_col]],
            s=80,
            facecolors="none",
            edgecolors="C3",
            linewidths=1.5,
            label=f"deepest @ k={int(sub.loc[idx, step_col])}",
        )
    ax.set_xlabel("Behavioral forgetting (learned − control Δ log-odds)")
    ax.set_ylabel("Relearning savings (baseline-adjusted early contrast)")
    ax.set_title("Savings vs forgetting")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_savings_vs_stream_step(
    frames: pd.DataFrame | Mapping[int, pd.DataFrame],
    *,
    out_path: str | Path,
    drop_early: bool = False,
    early_steps: tuple[int, ...] = (0, 25),
    step_col: str = "stream_steps",
    dlo_col: str = "delta_log_odds",
    y_col: str = "early_contrast",
    ci_col: str = "early_contrast_ci",
) -> Path:
    """Dose-response: forgetting and savings versus Phase-B stream step."""
    if isinstance(frames, pd.DataFrame):
        seed_map: dict[int, pd.DataFrame] = {-1: frames}
    else:
        seed_map = {int(k): v.copy() for k, v in frames.items()}

    cleaned: dict[int, pd.DataFrame] = {}
    for seed, df in seed_map.items():
        sub = df.dropna(subset=[step_col, y_col]).copy()
        if drop_early and step_col in sub.columns:
            sub = sub[~sub[step_col].isin(list(early_steps))]
        sub = sub.sort_values(step_col)
        if not sub.empty:
            cleaned[seed] = sub
    out = Path(out_path)
    ensure_dir(out.parent)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)

    ax = axes[0]
    ax.axhline(0.0, color="0.5", lw=1, ls="--")
    for i, (seed, sub) in enumerate(sorted(cleaned.items())):
        label = "Δ log-odds" if seed < 0 else f"seed {seed}"
        ax.plot(sub[step_col], sub[dlo_col], marker="o", color=f"C{i}", label=label)
        idx = sub[dlo_col].abs().idxmin()
        ax.axvline(float(sub.loc[idx, step_col]), color=f"C{i}", ls=":", alpha=0.5)
    if len(cleaned) > 1:
        steps = sorted(set().union(*[set(s[step_col]) for s in cleaned.values()]))
        means = []
        for step in steps:
            vals = [
                float(s.loc[s[step_col] == step, dlo_col].iloc[0])
                for s in cleaned.values()
                if (s[step_col] == step).any()
            ]
            means.append(float(np.mean(vals)) if vals else np.nan)
        ax.plot(steps, means, color="k", lw=2.0, marker="o", label="mean")
    ax.set_xlabel("Phase-B stream updates")
    ax.set_ylabel("Learned − control Δ log-odds")
    ax.set_title("A. Behavioral forgetting")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.axhline(0.0, color="0.5", lw=1, ls="--")
    for i, (seed, sub) in enumerate(sorted(cleaned.items())):
        label = "savings" if seed < 0 else f"seed {seed}"
        yerr = _ci_yerr(sub, y_col, ci_col)
        ax.errorbar(
            sub[step_col],
            sub[y_col],
            yerr=yerr,
            fmt="o-",
            color=f"C{i}",
            capsize=3,
            label=label,
        )
    if len(cleaned) > 1:
        steps = sorted(set().union(*[set(s[step_col]) for s in cleaned.values()]))
        means = []
        for step in steps:
            vals = [
                float(s.loc[s[step_col] == step, y_col].iloc[0])
                for s in cleaned.values()
                if (s[step_col] == step).any()
            ]
            means.append(float(np.mean(vals)) if vals else np.nan)
        ax.plot(steps, means, color="k", lw=2.0, marker="o", label="mean")
    ax.set_xlabel("Phase-B stream updates")
    ax.set_ylabel("Relearning savings (early contrast)")
    title = "B. Savings vs stream step"
    if drop_early:
        title += " (trough window)"
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_savings_vs_baseline(
    fact_df: pd.DataFrame,
    *,
    intercept: float,
    slope: float,
    out_path: str | Path,
    title: str = "Savings vs residual baseline difference",
) -> Path:
    """Scatter of fact-level s_i against d_i with OLS fit and intercept marked."""
    out = Path(out_path)
    ensure_dir(out.parent)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    if not fact_df.empty and "d_i" in fact_df.columns and "s_i" in fact_df.columns:
        ax.scatter(fact_df["d_i"], fact_df["s_i"], s=12, alpha=0.45, label="facts")
        x_line = np.linspace(
            float(fact_df["d_i"].min()), float(fact_df["d_i"].max()), 100
        )
        ax.plot(
            x_line,
            intercept + slope * x_line,
            color="C3",
            lw=2,
            label=f"fit: s = {intercept:.3f} + {slope:.3f}·d",
        )
        ax.axhline(0.0, color="0.5", lw=1, ls="--")
        ax.axvline(0.0, color="0.5", lw=1, ls="--")
        ax.scatter(
            [0.0],
            [intercept],
            s=80,
            facecolors="none",
            edgecolors="C3",
            linewidths=1.5,
            zorder=5,
            label=f"intercept @ d=0: {intercept:.3f}",
        )
    ax.set_xlabel("Baseline difference d (learned − control at k=0)")
    ax.set_ylabel("Fact-level savings s")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_gain_vs_baseline_curves(
    curves_df: pd.DataFrame,
    *,
    out_path: str | Path,
    primary_ks: tuple[int, ...] = (1, 2, 4, 8, 16),
    frac: float = 0.4,
) -> Path:
    """Fitted nonparametric gain-vs-baseline curves per relearn update k."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    out = Path(out_path)
    ensure_dir(out.parent)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    sub = curves_df[curves_df["k"].isin(list(primary_ks))].copy()
    for i, k in enumerate(primary_ks):
        g = sub[sub["k"] == k]
        if g.empty or "baseline_log_odds" not in g.columns:
            continue
        x = g["baseline_log_odds"].to_numpy(dtype=float)
        y = g["gain"].to_numpy(dtype=float)
        order = np.argsort(x)
        x_s, y_s = x[order], y[order]
        if np.unique(x_s).size < 3:
            continue
        fitted = lowess(y_s, x_s, frac=frac, return_sorted=True)
        ax.plot(fitted[:, 0], fitted[:, 1], color=f"C{i}", lw=2, label=f"k={k}")
    ax.axhline(0.0, color="0.5", lw=1, ls="--")
    ax.set_xlabel("Baseline target log-odds L₀")
    ax.set_ylabel("Gain (L_k − L₀)")
    ax.set_title("Gain vs baseline (LOWESS by relearn update)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_js_noise_floor(
    *,
    null_timeline: pd.DataFrame,
    cross_arm_timeline: pd.DataFrame | None = None,
    out_path: str | Path,
    pilot_margin: float = 0.03,
    confirmatory_margin: float = 0.02,
    measurement_floor: float | None = None,
) -> Path:
    """Overlay cross-seed same-history JS null on the cross-arm JS timeline."""
    out = Path(out_path)
    ensure_dir(out.parent)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if cross_arm_timeline is not None and not cross_arm_timeline.empty:
        ax.plot(
            cross_arm_timeline["stream_steps"],
            cross_arm_timeline["mean_js"],
            marker="o",
            color="C0",
            label="cross-arm JS (learned vs control)",
        )
    if not null_timeline.empty:
        ax.plot(
            null_timeline["stream_steps"],
            null_timeline["mean_js"],
            marker="s",
            color="C1",
            label="null: same-history cross-seed JS",
        )
        if "mean_js_ci" in null_timeline.columns:
            lowers, uppers = [], []
            for _, row in null_timeline.iterrows():
                ci = row.get("mean_js_ci")
                y = float(row["mean_js"])
                if isinstance(ci, (list, tuple)) and len(ci) == 2:
                    lowers.append(y - float(ci[0]))
                    uppers.append(float(ci[1]) - y)
                else:
                    lowers.append(0.0)
                    uppers.append(0.0)
            ax.fill_between(
                null_timeline["stream_steps"],
                null_timeline["mean_js"] - np.asarray(lowers),
                null_timeline["mean_js"] + np.asarray(uppers),
                color="C1",
                alpha=0.2,
            )
    ax.axhline(pilot_margin, color="C3", ls="--", label=f"pilot Δ_JS={pilot_margin}")
    ax.axhline(
        confirmatory_margin,
        color="C2",
        ls=":",
        label=f"confirmatory Δ_JS={confirmatory_margin}",
    )
    if measurement_floor is not None:
        ax.axhline(
            measurement_floor,
            color="0.4",
            ls="-.",
            label=f"measurement floor={measurement_floor:.3f}",
        )
    ax.set_xlabel("Phase-B stream updates")
    ax.set_ylabel("Jensen–Shannon distance")
    ax.set_title("JS noise floor vs cross-arm distance")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out
