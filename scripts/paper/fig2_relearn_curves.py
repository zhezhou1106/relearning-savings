"""Figure 2 — Relearning curves at k=100 (L/C/W/FRESH)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CRITERION_PROB,
    HISTORY_COLORS,
    HISTORY_LABELS,
    K_TROUGH,
    RELEARN_KS,
    SEEDS,
    apply_style,
    ensure_results_dir,
    load_curves_3arm_trough,
    load_fresh_curves,
    save_fig,
)


def _pool_curves() -> pd.DataFrame:
    parts = []
    for seed in SEEDS:
        df3 = load_curves_3arm_trough(seed)
        if not df3.empty:
            parts.append(df3)
        fresh = load_fresh_curves(seed)
        if not fresh.empty:
            fresh = fresh.copy()
            fresh["history"] = "fresh"
            parts.append(fresh)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    return df[df["k"].isin(RELEARN_KS)].copy()


def _pooled_with_boot_ci(
    df: pd.DataFrame, *, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for hist in ("learned", "control", "wrong", "fresh"):
        sub = df[df["history"] == hist]
        if sub.empty:
            continue
        unit = sub.groupby(["seed", "fact_id", "k"], as_index=False)["target_prob"].mean()
        for k in RELEARN_KS:
            uk = unit[unit["k"] == k]
            if uk.empty:
                continue
            vals = uk["target_prob"].to_numpy(dtype=float)
            point = float(vals.mean())
            n = len(vals)
            boots = np.empty(n_boot, dtype=float)
            for i in range(n_boot):
                boots[i] = float(vals[rng.integers(0, n, size=n)].mean())
            lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
            rows.append(
                {
                    "history": hist,
                    "k": int(k),
                    "mean": point,
                    "ci_lo": float(lo),
                    "ci_hi": float(hi),
                }
            )
    return pd.DataFrame(rows)


def _median_crossing(curve: pd.DataFrame, criterion: float = CRITERION_PROB) -> float | None:
    sub = curve.sort_values("k")
    ks = sub["k"].to_numpy(dtype=float)
    ps = sub["mean"].to_numpy(dtype=float)
    if len(ps) == 0:
        return None
    if ps[0] >= criterion:
        return 0.0
    for i in range(1, len(ks)):
        if ps[i] >= criterion and ps[i - 1] < criterion:
            t = (criterion - ps[i - 1]) / (ps[i] - ps[i - 1] + 1e-12)
            return float(ks[i - 1] + t * (ks[i] - ks[i - 1]))
    return None


def _x_plot(k: np.ndarray) -> np.ndarray:
    return np.where(k == 0, 0.5, k.astype(float))


def _style_relearn_xaxis(ax: plt.Axes) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.4, 80)
    ax.set_xticks([0.5, 1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["0", "1", "2", "4", "8", "16", "32", "64"])
    ax.set_xlabel("Relearning step u")


def _compute_crossings(stats: pd.DataFrame) -> dict[str, float | None]:
    crossings: dict[str, float | None] = {}
    for hist in ("learned", "control", "wrong", "fresh"):
        sub = stats[stats["history"] == hist].sort_values("k")
        if sub.empty:
            continue
        crossings[hist] = _median_crossing(sub)
    return crossings


def plot_panel_a(
    stats: pd.DataFrame,
    crossings: dict[str, float | None] | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Axes, dict[str, float | None]]:
    """Main relearning curves panel."""
    standalone = ax is None
    if standalone:
        apply_style()
        _, ax = plt.subplots(figsize=(5.8, 3.6))

    if crossings is None:
        crossings = _compute_crossings(stats)

    for hist in ("learned", "control", "wrong", "fresh"):
        sub = stats[stats["history"] == hist].sort_values("k")
        if sub.empty:
            continue
        x = _x_plot(sub["k"].to_numpy())
        color = HISTORY_COLORS[hist]
        ax.plot(
            x,
            sub["mean"],
            color=color,
            lw=1.7,
            marker="o",
            markersize=3.5,
            label=HISTORY_LABELS[hist],
        )
        ax.fill_between(x, sub["ci_lo"], sub["ci_hi"], color=color, alpha=0.18, linewidth=0)

    ax.axhline(CRITERION_PROB, color="0.4", ls="--", lw=0.9, label="criterion 0.9")
    ax.axvspan(_x_plot(np.array([1]))[0], 16, color="0.88", alpha=0.55, zorder=0)
    ax.text(4, 0.05, "early window U", fontsize=7, color="0.35", ha="center")

    y_mark = CRITERION_PROB
    for hist, u in crossings.items():
        if u is None:
            continue
        ax.plot(
            _x_plot(np.array([u]))[0],
            y_mark,
            marker="|",
            color=HISTORY_COLORS[hist],
            ms=14,
            mew=2,
        )

    u_l = crossings.get("learned")
    u_c = crossings.get("control")
    u_f = crossings.get("fresh")
    if u_l is not None and u_c is not None:
        ax.annotate(
            f"L→C: {u_c - u_l:.1f} upd",
            xy=(_x_plot(np.array([(u_l + u_c) / 2]))[0], y_mark + 0.03),
            fontsize=7,
            ha="center",
            color="0.25",
        )
    if u_l is not None and u_f is not None:
        ax.annotate(
            f"L→FRESH: {u_f - u_l:.1f} upd",
            xy=(_x_plot(np.array([(u_l + u_f) / 2]))[0], y_mark - 0.08),
            fontsize=7,
            ha="center",
            color="0.25",
        )

    ax.set_ylabel(r"Mean $\tilde{p}(a\mid e,r)$")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"(a) Relearning at trough (k={K_TROUGH})")
    ax.legend(frameon=False, loc="lower right", ncol=2)
    if standalone:
        _style_relearn_xaxis(ax)
        ax.figure.tight_layout()
    else:
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.4, 80)
    return ax, crossings


def plot_panel_b(stats: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Difference curve $\\bar{p}^L - \\bar{p}^C$."""
    standalone = ax is None
    if standalone:
        apply_style()
        # Match panel (a) height; same y-tick set, just more vertical space between ticks.
        _, ax = plt.subplots(figsize=(5.8, 3.6))

    learned = stats[stats["history"] == "learned"].set_index("k")["mean"]
    control = stats[stats["history"] == "control"].set_index("k")["mean"]
    common = sorted(set(learned.index) & set(control.index))
    if common:
        diff = learned.loc[common].to_numpy() - control.loc[common].to_numpy()
        ax.plot(
            _x_plot(np.array(common)),
            diff,
            color="#1f77b4",
            lw=1.5,
            marker="o",
            markersize=3,
        )
        ax.axhline(0.0, color="0.5", ls="--", lw=0.8)
        ax.axvspan(_x_plot(np.array([1]))[0], 16, color="0.88", alpha=0.55, zorder=0)
    ax.set_ylabel(r"$\bar{p}^L-\bar{p}^C$")
    ax.set_title("(b) Per-step LEARNED−CONTROL gap")
    # Same tick count as the short panel; taller figsize only spreads them apart.
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6, prune=None))
    _style_relearn_xaxis(ax)
    if standalone:
        ax.figure.tight_layout()
    return ax


def plot_figure2(stats: pd.DataFrame) -> tuple[Path, dict[str, float | None]]:
    apply_style()
    fig, (ax, ax_d) = plt.subplots(
        2,
        1,
        figsize=(5.8, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0]},
    )
    _, crossings = plot_panel_a(stats, ax=ax)
    plot_panel_b(stats, ax=ax_d)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelbottom=False)
    ax_d.set_title("")
    fig.tight_layout()
    return save_fig(fig, "fig2_relearn_curves_k100.pdf"), crossings


def plot_figure2_panels(stats: pd.DataFrame) -> tuple[Path, Path, dict[str, float | None]]:
    apply_style()
    _, crossings = plot_panel_a(stats)
    out_a = save_fig(plt.gcf(), "fig2a_relearn_curves.png")

    apply_style()
    plot_panel_b(stats)
    out_b = save_fig(plt.gcf(), "fig2b_lc_gap.png")
    return out_a, out_b, crossings


def main() -> tuple[Path, Path, Path]:
    ensure_results_dir()
    df = _pool_curves()
    if df.empty:
        raise SystemExit("No relearn curves found for Figure 2")
    stats = _pooled_with_boot_ci(df)
    out_pdf, crossings = plot_figure2(stats)
    out_a, out_b, _ = plot_figure2_panels(stats)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_a}")
    print(f"Wrote {out_b}")
    print("Crossings:", {h: (None if v is None else round(v, 3)) for h, v in crossings.items()})
    return out_pdf, out_a, out_b


if __name__ == "__main__":
    main()
