"""Figure 1 — Behavioral timeline: accessibility + ΔL/JS gap."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CHANCE_EM,
    HISTORY_COLORS,
    HISTORY_LABELS,
    JS_MARGIN,
    K_TROUGH,
    SEEDS,
    STREAM_GRID,
    apply_style,
    ensure_results_dir,
    load_js_noise_floor,
    load_manifest_3arm,
    save_fig,
    timeline_on_grid,
)


def _arm_series_from_timeline(rows: list[dict]) -> pd.DataFrame:
    """Extract L/C/W EM, L/C LO, ΔL, JS per stream step for one seed."""
    out = []
    for row in rows:
        k = int(row["stream_steps"])
        lvc = row.get("learned_vs_control") or row
        lvw = row.get("learned_vs_wrong") or {}
        wrong_em = float(lvw.get("control_exact_match", np.nan))
        wrong_lo = float(lvw.get("control_log_odds", np.nan))
        out.append(
            {
                "stream_steps": k,
                "learned_em": float(lvc.get("learned_exact_match", row.get("learned_exact_match"))),
                "control_em": float(lvc.get("control_exact_match", row.get("control_exact_match"))),
                "wrong_em": wrong_em,
                "learned_lo": float(lvc.get("learned_log_odds", row.get("learned_log_odds"))),
                "control_lo": float(lvc.get("control_log_odds", row.get("control_log_odds"))),
                "wrong_lo": wrong_lo,
                "delta_log_odds": float(lvc.get("delta_log_odds", row.get("delta_log_odds"))),
                "mean_js": float(lvc.get("mean_js", row.get("mean_js"))),
            }
        )
    return pd.DataFrame(out)


def collect_seed_timelines() -> pd.DataFrame:
    parts = []
    for seed in SEEDS:
        rows = timeline_on_grid(load_manifest_3arm(seed), STREAM_GRID)
        df = _arm_series_from_timeline(rows)
        df["seed"] = seed
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def _mean_sd(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby("stream_steps")[col]
    return pd.DataFrame(
        {
            "stream_steps": g.mean().index.astype(int),
            "mean": g.mean().values,
            "sd": g.std(ddof=1).values,
        }
    )


def _style_stream_axis(ax: plt.Axes, *, show_xlabel: bool) -> None:
    ax.set_xticks(list(STREAM_GRID))
    ax.tick_params(axis="x", labelrotation=45)
    if show_xlabel:
        ax.set_xlabel("Stream step k")


def plot_panel_a(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Panel (a): absolute exact-match rate by arm."""
    standalone = ax is None
    if standalone:
        apply_style()
        _, ax = plt.subplots(figsize=(4.2, 2.8))

    for hist, col in (
        ("learned", "learned_em"),
        ("control", "control_em"),
        ("wrong", "wrong_em"),
    ):
        stats = _mean_sd(df, col)
        color = HISTORY_COLORS[hist]
        ax.plot(
            stats["stream_steps"],
            stats["mean"],
            color=color,
            lw=1.6,
            marker="o",
            markersize=3.5,
            label=HISTORY_LABELS[hist],
        )
        ax.fill_between(
            stats["stream_steps"],
            stats["mean"] - stats["sd"],
            stats["mean"] + stats["sd"],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
    ax.axhline(CHANCE_EM, color="0.45", ls="--", lw=0.9, label="chance 1/8")
    ax.axvline(K_TROUGH, color="0.35", ls=":", lw=1.0)
    ax.text(
        K_TROUGH,
        0.98,
        " confirmatory\n checkpoint",
        transform=ax.get_xaxis_transform(),
        fontsize=7,
        color="0.3",
        va="top",
        ha="left",
    )
    ax.set_ylabel("Exact-match rate")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("(a) Accessibility (absolute)")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    _style_stream_axis(ax, show_xlabel=standalone)

    if standalone:
        ax.figure.tight_layout()
    return ax


def plot_panel_b(df: pd.DataFrame, js_null: float, ax: plt.Axes | None = None) -> plt.Axes:
    """Panel (b): ΔL and JS distance with margin/null annotations."""
    standalone = ax is None
    if standalone:
        apply_style()
        _, ax = plt.subplots(figsize=(4.2, 2.8))

    dlo = _mean_sd(df, "delta_log_odds")
    js = _mean_sd(df, "mean_js")
    ax.axvspan(75, 125, color="0.88", alpha=0.55, zorder=0, label="trough region")
    ax.plot(
        dlo["stream_steps"],
        dlo["mean"],
        color="#1f77b4",
        lw=1.6,
        marker="o",
        markersize=3.5,
        label="ΔL (LEARNED−CONTROL)",
    )
    ax.fill_between(
        dlo["stream_steps"],
        dlo["mean"] - dlo["sd"],
        dlo["mean"] + dlo["sd"],
        color="#1f77b4",
        alpha=0.18,
        linewidth=0,
    )
    ax.set_ylabel("ΔL (nats)")
    ax.axvline(K_TROUGH, color="0.35", ls=":", lw=1.0)

    ax_js = ax.twinx()
    ax_js.plot(
        js["stream_steps"],
        js["mean"],
        color="#9467bd",
        lw=1.6,
        ls="--",
        marker="s",
        markersize=3.2,
        label="JS distance",
    )
    ax_js.fill_between(
        js["stream_steps"],
        js["mean"] - js["sd"],
        js["mean"] + js["sd"],
        color="#9467bd",
        alpha=0.15,
        linewidth=0,
    )
    ax_js.axhspan(0.0, JS_MARGIN, color="#2ca02c", alpha=0.15, label=f"JS margin {JS_MARGIN}")
    ax_js.axhline(
        js_null,
        color="#d62728",
        ls="-.",
        lw=1.2,
        label=f"same-history null {js_null:.3f}",
    )
    ax_js.set_ylabel("JS distance")

    y0 = min(0.0, float((dlo["mean"] - dlo["sd"]).min()) - 0.05)
    y1 = float((dlo["mean"] + dlo["sd"]).max()) + 0.1
    ax.set_ylim(y0, y1)

    ax.set_title("(b) Gap and distribution distance")
    _style_stream_axis(ax, show_xlabel=standalone)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_js.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right", fontsize=7)

    if standalone:
        ax.figure.tight_layout()
    return ax


def plot_figure1(df: pd.DataFrame, js_null: float) -> Path:
    apply_style()
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(4.2, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.05, 1.0]}
    )
    plot_panel_a(df, ax_a)
    plot_panel_b(df, js_null, ax_b)
    ax_a.set_xlabel("")
    ax_a.tick_params(axis="x", labelbottom=False)
    fig.tight_layout()
    return save_fig(fig, "fig1_behavioral_timeline.pdf")


def plot_figure1_panels(df: pd.DataFrame, js_null: float) -> tuple[Path, Path]:
    """Standalone PNG exports for panels (a) and (b)."""
    apply_style()
    plot_panel_a(df)
    out_a = save_fig(plt.gcf(), "fig1a_accessibility.png")

    apply_style()
    plot_panel_b(df, js_null)
    out_b = save_fig(plt.gcf(), "fig1b_gap_js.png")
    return out_a, out_b


def main() -> tuple[Path, Path, Path]:
    ensure_results_dir()
    df = collect_seed_timelines()
    js = load_js_noise_floor()
    js_null = float(js.get("null_min_mean_js", 0.040))
    out_pdf = plot_figure1(df, js_null)
    out_a, out_b = plot_figure1_panels(df, js_null)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_a}")
    print(f"Wrote {out_b}")
    return out_pdf, out_a, out_b


if __name__ == "__main__":
    main()
