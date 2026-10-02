"""Shared paths, loaders, and helpers for the paper figures and analyses."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "artifacts" / "paper"
SCRIPTS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis.hierarchical import (  # noqa: E402
    PRIMARY_KS,
    fact_level_savings,
    fit_primary_contrast,
)
from util.paths import complementary_dir, eval_dirname, relearn_stream_dir  # noqa: E402

SEEDS = (0, 1, 2)
STREAM_GRID = (0, 25, 50, 75, 100, 125, 150, 200, 250, 400)
K_TROUGH = 100
PHASE_B_TAG = "dense_early"
RELEARN_TAG = "dense_early_slow1e5"
FRESH_TAG = "dense_early_slow1e5_fresh"
SPLIT = "confirmatory"
EXP_2ARM = "phase1"
EXP_3ARM = "phase1_3arm"
CHANCE_EM = 1.0 / 8.0
CHANCE_LO = float(np.log(1.0 / 7.0))  # ≈ -1.945910
JS_MARGIN = 0.03
CRITERION_PROB = 0.9
RELEARN_KS = (0, 1, 2, 4, 8, 16, 32, 64)

CONTRAST_ORDER = ("learned_vs_control", "learned_vs_wrong", "wrong_vs_control")
CONTRAST_LABELS = {
    "learned_vs_control": "LEARNED−CONTROL",
    "learned_vs_wrong": "LEARNED−WRONG",
    "wrong_vs_control": "WRONG−CONTROL",
}
HISTORY_COLORS = {
    "learned": "#1f77b4",
    "control": "#ff7f0e",
    "wrong": "#2ca02c",
    "fresh": "#d62728",
}
HISTORY_LABELS = {
    "learned": "LEARNED",
    "control": "CONTROL",
    "wrong": "WRONG",
    "fresh": "FRESH",
}


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: Path | str, payload: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    return out


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def save_fig(fig: plt.Figure, name: str) -> Path:
    ensure_results_dir()
    out = RESULTS_DIR / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def confirmatory_ids() -> set[str]:
    splits = read_json(REPO_ROOT / "data" / "facts" / "splits.json")
    return {str(x) for x in splits[SPLIT]}


def discover_runs(exp: str, seed: int) -> list[int]:
    seed_root = complementary_dir(exp, seed, 0).parent
    runs = []
    for p in sorted(seed_root.glob("c*")):
        if p.is_dir() and p.name.startswith("c") and p.name[1:].isdigit():
            runs.append(int(p.name[1:]))
    return runs or [0, 1]


def load_manifest_3arm(seed: int) -> dict:
    path = (
        REPO_ROOT
        / "data"
        / "manifests_3arm"
        / f"selected_checkpoints_seed_{seed}_{PHASE_B_TAG}.json"
    )
    return read_json(path)


def load_manifest_2arm(seed: int) -> dict:
    path = (
        REPO_ROOT
        / "data"
        / "manifests"
        / f"selected_checkpoints_seed_{seed}_{PHASE_B_TAG}.json"
    )
    return read_json(path)


def load_js_noise_floor() -> dict:
    return read_json(REPO_ROOT / "artifacts" / "analysis" / "js_noise_floor_dense_early.json")


def load_savings_2arm(seed: int) -> dict:
    path = (
        REPO_ROOT
        / "artifacts"
        / "analysis"
        / "p5"
        / f"savings_vs_forgetting_seed_{seed}_{RELEARN_TAG}_{SPLIT}.json"
    )
    return read_json(path)


def load_savings_3arm(seed: int) -> dict:
    path = (
        REPO_ROOT
        / "artifacts"
        / "analysis"
        / "p5"
        / (
            f"savings_vs_forgetting_seed_{seed}_{RELEARN_TAG}_{SPLIT}"
            f"_learned-vs-control_learned-vs-wrong_wrong-vs-control.json"
        )
    )
    return read_json(path)


def timeline_on_grid(manifest: dict, grid: Iterable[int] = STREAM_GRID) -> list[dict]:
    want = set(int(x) for x in grid)
    rows = []
    for row in manifest.get("timeline") or []:
        k = int(row["stream_steps"])
        if k in want:
            rows.append(row)
    rows.sort(key=lambda r: int(r["stream_steps"]))
    return rows


def _curves_at_step(
    exp: str,
    seed: int,
    tag: str,
    step: int,
    split: str,
    runs: list[int] | None = None,
) -> pd.DataFrame:
    parts = []
    for comp in runs or discover_runs(exp, seed):
        path = relearn_stream_dir(exp, seed, comp, tag, step, split) / "curves.parquet"
        if path.exists():
            parts.append(pd.read_parquet(path))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def load_curves_3arm_trough(seed: int) -> pd.DataFrame:
    keep = confirmatory_ids()
    df = _curves_at_step(EXP_3ARM, seed, RELEARN_TAG, K_TROUGH, SPLIT)
    if df.empty:
        return df
    return df[df["fact_id"].astype(str).isin(keep)].copy()


def load_curves_2arm_trough(seed: int) -> pd.DataFrame:
    keep = confirmatory_ids()
    df = _curves_at_step(EXP_2ARM, seed, RELEARN_TAG, K_TROUGH, SPLIT)
    if df.empty:
        return df
    return df[df["fact_id"].astype(str).isin(keep)].copy()


def load_fresh_curves(seed: int) -> pd.DataFrame:
    parts = []
    for comp in discover_runs(EXP_2ARM, seed):
        path = (
            complementary_dir(EXP_2ARM, seed, comp)
            / "relearn"
            / FRESH_TAG
            / "fresh_pool"
            / f"stream_{K_TROUGH}"
            / f"c{comp}"
            / "curves.parquet"
        )
        if path.exists():
            parts.append(pd.read_parquet(path))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def behavioral_baseline_deltas(
    *,
    exp: str,
    seed: int,
    step: int,
    tag: str,
    fact_ids: set[str],
    treatment: str,
    reference: str,
) -> dict[str, float]:
    by_arm: dict[str, dict[str, float]] = {treatment: {}, reference: {}}
    for comp in discover_runs(exp, seed):
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
    return {fid: by_arm[treatment][fid] - by_arm[reference][fid] for fid in common}


def fact_savings_at_trough(
    seed: int,
    *,
    exp: str = EXP_2ARM,
    treatment: str = "learned",
    reference: str = "control",
) -> pd.DataFrame:
    keep = confirmatory_ids()
    curves = _curves_at_step(exp, seed, RELEARN_TAG, K_TROUGH, SPLIT)
    if curves.empty:
        return pd.DataFrame(columns=["fact_id", "s_i", "d_i", "seed"])
    curves = curves[curves["fact_id"].astype(str).isin(keep)].copy()
    curves["checkpoint_role"] = "forgetting_sweep"
    beh = behavioral_baseline_deltas(
        exp=exp,
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
    fact["seed"] = seed
    return fact


def ols_with_ci(
    x: np.ndarray, y: np.ndarray, *, alpha: float = 0.05
) -> dict[str, float]:
    """Simple OLS y = a + b x with analytic 95% CI for slope (large-n t approx)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(x.size)
    if n < 3:
        return {
            "intercept": float("nan"),
            "slope": float("nan"),
            "slope_se": float("nan"),
            "slope_ci_lo": float("nan"),
            "slope_ci_hi": float("nan"),
            "n": n,
        }
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    resid = y - X @ coef
    dof = n - 2
    s2 = float(np.sum(resid**2) / dof)
    xtx_inv = np.linalg.inv(X.T @ X)
    se_b = float(np.sqrt(s2 * xtx_inv[1, 1]))
    # t critical ~1.96 for large n; use scipy if available
    try:
        from scipy import stats

        tcrit = float(stats.t.ppf(1 - alpha / 2, dof))
    except Exception:
        tcrit = 1.96
    return {
        "intercept": a,
        "slope": b,
        "slope_se": se_b,
        "slope_ci_lo": b - tcrit * se_b,
        "slope_ci_hi": b + tcrit * se_b,
        "n": n,
        "tcrit": tcrit,
    }


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 200,
        }
    )


__all__ = [
    "PRIMARY_KS",
    "fit_primary_contrast",
    "fact_level_savings",
]
