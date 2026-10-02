"""Hierarchical relearning analysis for the workshop primary estimand.

Primary estimand: baseline-adjusted learned-minus-control contrast averaged
over early updates k in {1, 2, 4, 8, 16}.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

PRIMARY_ROLE = "behavioral_equivalence"
PRIMARY_KS = (1, 2, 4, 8, 16)


def paired_curve_difference(
    df: pd.DataFrame,
    *,
    treatment: str = "learned",
    reference: str = "control",
    checkpoint_role: str | None = PRIMARY_ROLE,
) -> pd.DataFrame:
    """Per fact and k, treatment gain minus reference gain (secondary / power)."""
    sub = df[df["history"].isin([treatment, reference])].copy()
    if checkpoint_role is not None and "checkpoint_role" in sub.columns:
        sub = sub[sub["checkpoint_role"] == checkpoint_role]
    if sub.empty:
        return pd.DataFrame(columns=["fact_id", "k", "log_k", "delta_gain"])
    sub = add_design_columns(sub)
    wide = (
        sub.pivot_table(
            index=["fact_id", "k", "log_k"],
            columns="history",
            values="gain",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    if treatment not in wide.columns or reference not in wide.columns:
        return pd.DataFrame(columns=["fact_id", "k", "log_k", "delta_gain"])
    wide = wide.dropna(subset=[treatment, reference])
    wide["delta_gain"] = wide[treatment] - wide[reference]
    return wide[["fact_id", "k", "log_k", "delta_gain"]]


def add_design_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_k"] = np.log2(out["k"].astype(float) + 1.0)
    out["history_learned"] = (out["history"] == "learned").astype(int)
    return out


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def paired_absolute_levels(
    df: pd.DataFrame,
    *,
    treatment: str = "learned",
    reference: str = "control",
    checkpoint_role: str | None = PRIMARY_ROLE,
    ks: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Per fact and k: treatment and control absolute target log-odds + baseline."""
    sub = df[df["history"].isin([treatment, reference])].copy()
    if checkpoint_role is not None and "checkpoint_role" in sub.columns:
        sub = sub[sub["checkpoint_role"] == checkpoint_role]
    if ks is not None:
        sub = sub[sub["k"].isin(list(ks))]
    if sub.empty:
        return pd.DataFrame()

    # Baseline = k=0 absolute LO for the same fact×history.
    base = df[df["history"].isin([treatment, reference])].copy()
    if "k" in base.columns:
        base = base[base["k"] == 0]
    else:
        base = base.iloc[0:0]
    if checkpoint_role is not None and "checkpoint_role" in base.columns:
        base = base[base["checkpoint_role"] == checkpoint_role]
    base_map = {
        (str(r.fact_id), str(r.history)): float(r.target_log_odds)
        for r in base.itertuples(index=False)
        if hasattr(r, "target_log_odds")
    }

    rows = []
    for (fact_id, k), g in sub.groupby(["fact_id", "k"], sort=True):
        by_h = {r.history: r for r in g.itertuples(index=False)}
        if treatment not in by_h or reference not in by_h:
            continue
        t = by_h[treatment]
        c = by_h[reference]
        t_base = base_map.get(
            (str(fact_id), treatment),
            float(getattr(t, "baseline_log_odds", t.target_log_odds)),
        )
        c_base = base_map.get(
            (str(fact_id), reference),
            float(getattr(c, "baseline_log_odds", c.target_log_odds)),
        )
        rows.append(
            {
                "fact_id": fact_id,
                "k": int(k),
                "learned_lo": float(t.target_log_odds),
                "control_lo": float(c.target_log_odds),
                "learned_baseline": t_base,
                "control_baseline": c_base,
                "delta_lo": float(t.target_log_odds) - float(c.target_log_odds),
                "delta_baseline": t_base - c_base,
                "delta_gain": float(getattr(t, "gain", t.target_log_odds - t_base))
                - float(getattr(c, "gain", c.target_log_odds - c_base)),
            }
        )
    return pd.DataFrame(rows)


def fit_primary_contrast(
    df: pd.DataFrame,
    *,
    treatment: str = "learned",
    reference: str = "control",
    checkpoint_role: str | None = PRIMARY_ROLE,
    primary_ks: tuple[int, ...] = PRIMARY_KS,
) -> dict[str, float]:
    """Baseline-adjusted early-update contrast (workshop primary estimand).

    For each fact and early k, residualize absolute LO on the fact's k=0 LO
    (within history), then average the learned-minus-control residual.
    Practically: mean(delta_lo - delta_baseline) over early k, which equals the
    mean baseline-subtracted gain contrast.
    """
    paired = paired_absolute_levels(
        df,
        treatment=treatment,
        reference=reference,
        checkpoint_role=checkpoint_role,
        ks=primary_ks,
    )
    empty = {
        "early_contrast": 0.0,
        "integrated_gain": 0.0,
        "n_facts": 0,
        "n_obs": 0,
    }
    if paired.empty:
        return empty

    # Baseline-adjusted contrast: (LO_L - LO_C) - (base_L - base_C)
    paired = paired.copy()
    paired["adj_contrast"] = paired["delta_lo"] - paired["delta_baseline"]
    early = float(paired["adj_contrast"].mean())

    # Full-curve integrated summary on delta_gain vs log2(k+1) for secondary check.
    full = paired_absolute_levels(
        df,
        treatment=treatment,
        reference=reference,
        checkpoint_role=checkpoint_role,
        ks=None,
    )
    integrated = 0.0
    if not full.empty:
        full = full.copy()
        full["log_k"] = np.log2(full["k"].astype(float) + 1.0)
        grid = np.sort(full["log_k"].unique())
        means = [
            float(full.loc[full["log_k"] == x, "delta_gain"].mean()) if (full["log_k"] == x).any() else 0.0
            for x in grid
        ]
        if len(grid) >= 2:
            integrated = _trapezoid(np.asarray(means), grid)

    return {
        "early_contrast": early,
        "history_by_step": early,  # alias for older callers
        "integrated_gain": float(integrated),
        "n_facts": int(paired["fact_id"].nunique()),
        "n_obs": int(len(paired)),
    }


def counterbalanced_bootstrap_ci(
    df: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    treatment: str = "learned",
    reference: str = "control",
    checkpoint_role: str | None = PRIMARY_ROLE,
    primary_ks: tuple[int, ...] = PRIMARY_KS,
) -> dict[str, Any]:
    """Bootstrap over facts, preserving learned/control pairing."""
    rng = np.random.default_rng(seed)
    point = fit_primary_contrast(
        df,
        treatment=treatment,
        reference=reference,
        checkpoint_role=checkpoint_role,
        primary_ks=primary_ks,
    )
    paired = paired_absolute_levels(
        df,
        treatment=treatment,
        reference=reference,
        checkpoint_role=checkpoint_role,
        ks=primary_ks,
    )
    result: dict[str, Any] = {
        "early_contrast": point["early_contrast"],
        "history_by_step": point["early_contrast"],
        "integrated_gain": point["integrated_gain"],
        "n_facts": point["n_facts"],
        "n_obs": point["n_obs"],
        "n_boot": n_boot,
        "treatment": treatment,
        "reference": reference,
        "checkpoint_role": checkpoint_role,
        "primary_ks": list(primary_ks),
    }
    if paired.empty:
        result.update(
            {
                "early_contrast_ci": [0.0, 0.0],
                "history_by_step_ci": [0.0, 0.0],
                "integrated_gain_ci": [0.0, 0.0],
                "primary_pass": False,
                "warning": "no_paired_curves",
            }
        )
        return result

    paired = paired.copy()
    paired["adj_contrast"] = paired["delta_lo"] - paired["delta_baseline"]
    by_fact = {fid: g for fid, g in paired.groupby("fact_id", sort=True)}
    fact_ids = np.array(sorted(by_fact))
    contrasts = np.empty(n_boot, dtype=float)
    for i in tqdm(range(n_boot), desc="bootstrap CI"):
        sample = rng.choice(fact_ids, size=len(fact_ids), replace=True)
        boot = pd.concat([by_fact[f] for f in sample], ignore_index=True)
        contrasts[i] = float(boot["adj_contrast"].mean())

    lo, hi = alpha / 2, 1 - alpha / 2
    ci = [float(np.quantile(contrasts, lo)), float(np.quantile(contrasts, hi))]
    result.update(
        {
            "early_contrast_ci": ci,
            "history_by_step_ci": ci,
            "integrated_gain_ci": [point["integrated_gain"], point["integrated_gain"]],
            "primary_pass": bool(ci[0] > 0 and point["early_contrast"] > 0),
        }
    )
    return result


def baseline_residual_sensitivity(
    df: pd.DataFrame,
    *,
    checkpoint_role: str | None = PRIMARY_ROLE,
    primary_ks: tuple[int, ...] = PRIMARY_KS,
    central_band: float = 0.05,
) -> dict[str, Any]:
    """Association of baseline residuals with relearning advantage + narrow band."""
    paired = paired_absolute_levels(
        df,
        checkpoint_role=checkpoint_role,
        ks=primary_ks,
    )
    if paired.empty:
        return {"n_facts": 0, "correlation": 0.0, "narrow_band_contrast": 0.0}
    paired = paired.copy()
    paired["adj_contrast"] = paired["delta_lo"] - paired["delta_baseline"]
    fact = paired.groupby("fact_id", as_index=False).agg(
        delta_baseline=("delta_baseline", "mean"),
        adj_contrast=("adj_contrast", "mean"),
    )
    corr = (
        float(np.corrcoef(fact["delta_baseline"], fact["adj_contrast"])[0, 1])
        if len(fact) > 1
        else 0.0
    )
    if np.isnan(corr):
        corr = 0.0
    narrow = fact[fact["delta_baseline"].abs() <= central_band]
    narrow_contrast = float(narrow["adj_contrast"].mean()) if len(narrow) else 0.0
    return {
        "n_facts": int(len(fact)),
        "correlation": corr,
        "narrow_band": central_band,
        "n_narrow": int(len(narrow)),
        "narrow_band_contrast": narrow_contrast,
    }


def fact_level_savings(
    df: pd.DataFrame,
    *,
    treatment: str = "learned",
    reference: str = "control",
    checkpoint_role: str | None = PRIMARY_ROLE,
    primary_ks: tuple[int, ...] = PRIMARY_KS,
    baseline_deltas: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Per-fact savings s_i and baseline difference d_i.

    If ``baseline_deltas`` is provided (e.g. from the disjoint behavioral-eval
    template family), it replaces relearn-k0 ``delta_baseline`` as d_i to avoid
    errors-in-variables bias from sharing measurement noise with s_i.
    """
    paired = paired_absolute_levels(
        df,
        treatment=treatment,
        reference=reference,
        checkpoint_role=checkpoint_role,
        ks=primary_ks,
    )
    if paired.empty:
        return pd.DataFrame(columns=["fact_id", "s_i", "d_i"])
    paired = paired.copy()
    paired["adj_contrast"] = paired["delta_lo"] - paired["delta_baseline"]
    fact = paired.groupby("fact_id", as_index=False).agg(
        s_i=("adj_contrast", "mean"),
        d_i=("delta_baseline", "mean"),
    )
    if baseline_deltas is not None:
        fact["d_i"] = fact["fact_id"].map(
            lambda fid: float(baseline_deltas.get(str(fid), np.nan))
        )
        fact = fact.dropna(subset=["d_i"])
    return fact


def _ols_intercept_slope(d: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Ordinary least squares: s = a + b d. Returns (intercept, slope)."""
    if d.size < 2:
        return float(s.mean()) if s.size else 0.0, 0.0
    x = np.column_stack([np.ones(d.size), d])
    coef, *_ = np.linalg.lstsq(x, s, rcond=None)
    return float(coef[0]), float(coef[1])


def savings_baseline_regression(
    df: pd.DataFrame,
    *,
    treatment: str = "learned",
    reference: str = "control",
    checkpoint_role: str | None = PRIMARY_ROLE,
    primary_ks: tuple[int, ...] = PRIMARY_KS,
    baseline_deltas: dict[str, float] | None = None,
    d_source: str = "relearn_k0",
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Regress fact-level savings s_i on baseline difference d_i.

    Reports the intercept at d=0 (savings for a fact with zero residual
    behavioral difference) with a fact-bootstrap CI, plus the slope that
    settles the gain-vs-baseline confound empirically.
    """
    fact = fact_level_savings(
        df,
        treatment=treatment,
        reference=reference,
        checkpoint_role=checkpoint_role,
        primary_ks=primary_ks,
        baseline_deltas=baseline_deltas,
    )
    empty: dict[str, Any] = {
        "intercept": 0.0,
        "intercept_ci": [0.0, 0.0],
        "slope": 0.0,
        "slope_ci": [0.0, 0.0],
        "n_facts": 0,
        "primary_pass": False,
        "d_source": d_source,
        "n_boot": n_boot,
        "warning": "no_facts",
    }
    if fact.empty or len(fact) < 2:
        return empty

    d = fact["d_i"].to_numpy(dtype=float)
    s = fact["s_i"].to_numpy(dtype=float)
    intercept, slope = _ols_intercept_slope(d, s)

    rng = np.random.default_rng(seed)
    n = len(fact)
    boot_a = np.empty(n_boot, dtype=float)
    boot_b = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_a[i], boot_b[i] = _ols_intercept_slope(d[idx], s[idx])
    lo, hi = alpha / 2, 1 - alpha / 2
    a_ci = [float(np.quantile(boot_a, lo)), float(np.quantile(boot_a, hi))]
    b_ci = [float(np.quantile(boot_b, lo)), float(np.quantile(boot_b, hi))]
    return {
        "intercept": intercept,
        "intercept_ci": a_ci,
        "slope": slope,
        "slope_ci": b_ci,
        "n_facts": int(n),
        "primary_pass": bool(a_ci[0] > 0 and intercept > 0),
        "d_source": d_source,
        "n_boot": n_boot,
        "mean_s": float(s.mean()),
        "mean_d": float(d.mean()),
        "fact_level": fact,
    }


def pooled_gain_residual_contrast(
    all_curves_df: pd.DataFrame,
    *,
    treatment: str = "learned",
    reference: str = "control",
    primary_ks: tuple[int, ...] = PRIMARY_KS,
    checkpoint_role: str | None = "forgetting_sweep",
    stream_steps: int | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    frac: float = 0.4,
) -> dict[str, Any]:
    """Nonparametric residual contrast after fitting gain as a function of L_0.

    Pools arm-level observations across stream checkpoints to fit LOWESS
    g_hat_k(L_0) per relearn update k. Residuals are then contrasted between
    arms. If ``stream_steps`` is set, the contrast uses only that checkpoint
    (the fit still uses the full pool so the ceiling at k=0 anchors the curve).
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    sub = all_curves_df[all_curves_df["history"].isin([treatment, reference])].copy()
    if checkpoint_role is not None and "checkpoint_role" in sub.columns:
        sub = sub[sub["checkpoint_role"] == checkpoint_role]
    sub = sub[sub["k"].isin(list(primary_ks))]
    if sub.empty or "baseline_log_odds" not in sub.columns or "gain" not in sub.columns:
        return {
            "residual_contrast": 0.0,
            "residual_contrast_ci": [0.0, 0.0],
            "n_facts": 0,
            "primary_pass": False,
            "warning": "no_curves",
        }

    # Fit g_hat_k(L_0) on the pooled (all arms, all stream steps) cloud.
    residual_rows: list[dict[str, Any]] = []
    for k, g in sub.groupby("k", sort=True):
        x = g["baseline_log_odds"].to_numpy(dtype=float)
        y = g["gain"].to_numpy(dtype=float)
        if len(g) < 10 or np.unique(x).size < 3:
            pred = np.full(len(g), float(y.mean()) if len(y) else 0.0)
        else:
            fitted = lowess(y, x, frac=frac, return_sorted=False)
            pred = np.asarray(fitted, dtype=float)
        for row, r in zip(g.itertuples(index=False), y - pred):
            residual_rows.append(
                {
                    "fact_id": str(row.fact_id),
                    "history": str(row.history),
                    "k": int(k),
                    "stream_steps": int(getattr(row, "stream_steps", -1)),
                    "residual": float(r),
                }
            )
    resid = pd.DataFrame(residual_rows)
    if resid.empty:
        return {
            "residual_contrast": 0.0,
            "residual_contrast_ci": [0.0, 0.0],
            "n_facts": 0,
            "primary_pass": False,
            "warning": "empty_residuals",
        }

    if stream_steps is not None and "stream_steps" in resid.columns:
        resid = resid[resid["stream_steps"] == int(stream_steps)]

    # Per fact: mean residual under treatment minus mean residual under reference.
    wide = (
        resid.groupby(["fact_id", "history"], as_index=False)["residual"]
        .mean()
        .pivot(index="fact_id", columns="history", values="residual")
        .dropna()
    )
    if treatment not in wide.columns or reference not in wide.columns or wide.empty:
        return {
            "residual_contrast": 0.0,
            "residual_contrast_ci": [0.0, 0.0],
            "n_facts": 0,
            "primary_pass": False,
            "warning": "unpaired_residuals",
            "stream_steps": stream_steps,
        }
    contrasts = (wide[treatment] - wide[reference]).to_numpy(dtype=float)
    point = float(contrasts.mean())
    rng = np.random.default_rng(seed)
    draws = np.array(
        [
            float(rng.choice(contrasts, size=contrasts.size, replace=True).mean())
            for _ in range(n_boot)
        ]
    )
    lo, hi = alpha / 2, 1 - alpha / 2
    ci = [float(np.quantile(draws, lo)), float(np.quantile(draws, hi))]
    return {
        "residual_contrast": point,
        "residual_contrast_ci": ci,
        "n_facts": int(contrasts.size),
        "primary_pass": bool(ci[0] > 0 and point > 0),
        "n_boot": n_boot,
        "frac": frac,
        "stream_steps": stream_steps,
    }


def updates_to_threshold(
    df: pd.DataFrame,
    *,
    threshold: float,
    ks: tuple[int, ...] | list[int] | None = None,
    prob_col: str = "target_prob",
) -> pd.DataFrame:
    """Per fact × history: updates to reach target_prob >= threshold.

    Uses log2-linear interpolation between consecutive grid points. Right-censored
    observations (never reaching threshold) are marked with ``censored=True`` and
    ``updates`` equal to the last grid point (for KM estimation).
    """
    grid = list(ks) if ks is not None else sorted(df["k"].unique().tolist())
    grid = sorted(int(x) for x in grid)
    if not grid:
        return pd.DataFrame(columns=["fact_id", "history", "updates", "censored"])

    rows: list[dict[str, Any]] = []
    group_cols = ["fact_id", "history"]
    if "stream_steps" in df.columns:
        group_cols = ["fact_id", "history", "stream_steps"]
    for keys, g in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        g = g.sort_values("k")
        probs = {int(r.k): float(getattr(r, prob_col)) for r in g.itertuples(index=False)}
        # Already at threshold at k=0.
        p0 = probs.get(grid[0], float("nan"))
        if not np.isnan(p0) and p0 >= threshold:
            row = {
                "fact_id": str(keys[0]),
                "history": str(keys[1]),
                "updates": 0.0,
                "censored": False,
            }
            if len(keys) > 2:
                row["stream_steps"] = int(keys[2])
            rows.append(row)
            continue
        reached = False
        for i in range(1, len(grid)):
            k_lo, k_hi = grid[i - 1], grid[i]
            p_lo = probs.get(k_lo)
            p_hi = probs.get(k_hi)
            if p_lo is None or p_hi is None:
                continue
            if p_lo < threshold <= p_hi:
                # Log2-linear interpolation of the crossing.
                if p_hi == p_lo:
                    t = 1.0
                else:
                    t = (threshold - p_lo) / (p_hi - p_lo)
                log_lo = np.log2(k_lo + 1.0)
                log_hi = np.log2(k_hi + 1.0)
                updates = float(2 ** (log_lo + t * (log_hi - log_lo)) - 1.0)
                row = {
                    "fact_id": str(keys[0]),
                    "history": str(keys[1]),
                    "updates": updates,
                    "censored": False,
                }
                if len(keys) > 2:
                    row["stream_steps"] = int(keys[2])
                rows.append(row)
                reached = True
                break
        if not reached:
            row = {
                "fact_id": str(keys[0]),
                "history": str(keys[1]),
                "updates": float(grid[-1]),
                "censored": True,
            }
            if len(keys) > 2:
                row["stream_steps"] = int(keys[2])
            rows.append(row)
    return pd.DataFrame(rows)


def _km_median(times: np.ndarray, events: np.ndarray) -> float:
    """Kaplan–Meier median survival time; returns inf if median not reached."""
    order = np.argsort(times)
    times = times[order]
    events = events[order]
    n = len(times)
    if n == 0:
        return float("inf")
    unique_t = np.unique(times)
    surv = 1.0
    at_risk = n
    median = float("inf")
    for t in unique_t:
        died = int(events[times == t].sum())
        if at_risk > 0 and died > 0:
            surv *= 1.0 - died / at_risk
        if median == float("inf") and surv <= 0.5:
            median = float(t)
        at_risk -= int((times == t).sum())
    return median


def choose_threshold_on_development(
    df: pd.DataFrame,
    *,
    candidates: tuple[float, ...] = (0.90, 0.75, 0.60, 0.50, 0.40, 0.30),
    min_reach_rate: float = 0.70,
    treatment: str = "learned",
    reference: str = "control",
) -> dict[str, Any]:
    """Highest threshold where ≥ min_reach_rate of facts in both arms reach it."""
    chosen = None
    report = []
    for thr in candidates:
        utt = updates_to_threshold(df, threshold=thr)
        rates = {}
        for hist in (treatment, reference):
            sub = utt[utt["history"] == hist]
            rates[hist] = float((~sub["censored"]).mean()) if len(sub) else 0.0
        report.append({"threshold": thr, **rates})
        if rates.get(treatment, 0.0) >= min_reach_rate and rates.get(
            reference, 0.0
        ) >= min_reach_rate:
            chosen = thr
            break  # candidates are descending; first hit is highest
    # If none meet the bar, take the lowest candidate that maximizes min reach rate.
    if chosen is None and report:
        best = max(report, key=lambda r: min(r.get(treatment, 0.0), r.get(reference, 0.0)))
        chosen = float(best["threshold"])
    return {
        "threshold": chosen if chosen is not None else float(candidates[-1]),
        "min_reach_rate": min_reach_rate,
        "candidates": report,
    }


def threshold_savings(
    df: pd.DataFrame,
    *,
    threshold: float,
    treatment: str = "learned",
    reference: str = "control",
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    fresh_updates: pd.Series | None = None,
) -> dict[str, Any]:
    """Paired updates-to-threshold savings with KM medians and bootstrap CI.

    If ``fresh_updates`` (indexed by fact_id, or a scalar Series of fresh-arm
    times) is provided, also reports savings as a fraction of naive acquisition.
    """
    utt = updates_to_threshold(df, threshold=threshold)
    if utt.empty:
        return {
            "threshold": threshold,
            "treatment_median": float("inf"),
            "reference_median": float("inf"),
            "updates_saved": 0.0,
            "updates_saved_ci": [0.0, 0.0],
            "n_facts": 0,
            "primary_pass": False,
        }

    # Pair within fact across histories (and stream_steps when present).
    index_cols = ["fact_id"]
    if "stream_steps" in utt.columns:
        index_cols.append("stream_steps")
    wide_u = utt.pivot_table(
        index=index_cols, columns="history", values="updates", aggfunc="mean"
    )
    wide_c = utt.pivot_table(
        index=index_cols, columns="history", values="censored", aggfunc="max"
    )
    if treatment not in wide_u.columns or reference not in wide_u.columns:
        return {
            "threshold": threshold,
            "treatment_median": float("inf"),
            "reference_median": float("inf"),
            "updates_saved": 0.0,
            "updates_saved_ci": [0.0, 0.0],
            "n_facts": 0,
            "primary_pass": False,
            "warning": "unpaired",
        }
    common = wide_u.dropna(subset=[treatment, reference]).index
    t_times = wide_u.loc[common, treatment].to_numpy(dtype=float)
    r_times = wide_u.loc[common, reference].to_numpy(dtype=float)
    t_events = (~wide_c.loc[common, treatment].astype(bool)).to_numpy()
    r_events = (~wide_c.loc[common, reference].astype(bool)).to_numpy()
    # Paired difference: positive = treatment reaches threshold in fewer updates.
    paired_diff = r_times - t_times

    t_med = _km_median(t_times, t_events.astype(float))
    r_med = _km_median(r_times, r_events.astype(float))
    point = float(np.mean(paired_diff))

    rng = np.random.default_rng(seed)
    n = paired_diff.size
    draws = np.array(
        [
            float(rng.choice(paired_diff, size=n, replace=True).mean())
            for _ in range(n_boot)
        ]
    )
    lo, hi = alpha / 2, 1 - alpha / 2
    ci = [float(np.quantile(draws, lo)), float(np.quantile(draws, hi))]

    result: dict[str, Any] = {
        "threshold": threshold,
        "treatment_median": t_med,
        "reference_median": r_med,
        "updates_saved": point,
        "updates_saved_ci": ci,
        "n_facts": int(n),
        "treatment_reach_rate": float(t_events.mean()),
        "reference_reach_rate": float(r_events.mean()),
        "primary_pass": bool(ci[0] > 0 and point > 0),
        "n_boot": n_boot,
    }
    if fresh_updates is not None and len(fresh_updates):
        # Align by fact_id when possible (same panel). Fresh-pool facts are
        # disjoint, so fall back to the fresh-arm KM/median denominator.
        aligned: list[float] = []
        if hasattr(fresh_updates, "index") and len(common):
            fact_ids = (
                [c[0] for c in common]
                if isinstance(common[0], tuple)
                else list(common)
            )
            for i, fid in enumerate(fact_ids):
                if fid in fresh_updates.index:
                    denom = float(fresh_updates.loc[fid])
                    if denom > 0:
                        aligned.append(paired_diff[i] / denom)
        if aligned:
            frac = float(np.mean(aligned))
            frac_draws = np.array(
                [
                    float(rng.choice(aligned, size=len(aligned), replace=True).mean())
                    for _ in range(n_boot)
                ]
            )
            result["fraction_of_naive"] = frac
            result["fraction_of_naive_ci"] = [
                float(np.quantile(frac_draws, lo)),
                float(np.quantile(frac_draws, hi)),
            ]
            result["fraction_alignment"] = "per_fact"
        else:
            denom = float(np.median(np.asarray(fresh_updates, dtype=float)))
            if denom > 0:
                result["fraction_of_naive"] = point / denom
                result["fraction_of_naive_ci"] = [ci[0] / denom, ci[1] / denom]
                result["naive_median"] = denom
                result["fraction_alignment"] = "fresh_median"
    return result
