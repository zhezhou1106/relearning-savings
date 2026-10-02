"""Power calibration and equivalence-margin feasibility.

Both routines target the estimators the study actually uses: the paired
learned-minus-control slope for the primary hypothesis, and the two-one-sided
interval on the learned-minus-control gap for the equivalence gate.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.stats import norm
from tqdm import tqdm

from util.metrics import write_json

DEFAULT_KS = (0, 1, 2, 4, 8, 16, 32, 64)


def _cluster_robust_slope_se(
    x: np.ndarray, y: np.ndarray, cluster: np.ndarray
) -> tuple[float, float]:
    """OLS slope of y on x with a fact-clustered standard error.

    The analysis bootstraps over facts; the clustered sandwich SE is the
    asymptotic equivalent and is cheap enough to run inside a simulation loop.
    """
    design = np.column_stack([np.ones_like(x), x])
    xtx_inv = np.linalg.pinv(design.T @ design)
    coef = xtx_inv @ design.T @ y
    resid = y - design @ coef

    meat = np.zeros((2, 2))
    for c in np.unique(cluster):
        m = cluster == c
        xu = design[m].T @ resid[m]
        meat += np.outer(xu, xu)
    cov = xtx_inv @ meat @ xtx_inv
    return float(coef[1]), float(np.sqrt(max(cov[1, 1], 0.0)))


def simulate_power(
    *,
    n_facts: int = 192,
    n_steps: int = 8,
    effect_slope: float = 0.2,
    fact_sd: float = 0.5,
    noise_sd: float = 0.3,
    n_sims: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
    ks: Sequence[int] = DEFAULT_KS,
    effect_integrated_gain: float | None = None,
) -> dict[str, Any]:
    """Power for the paired learned-minus-control slope.

    Each simulated fact contributes a learned and a control curve sharing a fact
    level offset; the estimator is the pooled OLS slope of their difference on
    log2(k+1), tested against zero with a fact-clustered SE. That is the same
    quantity `fit_primary_contrast` reports.
    """
    if effect_integrated_gain is not None and effect_slope is None:
        effect_slope = effect_integrated_gain

    rng = np.random.default_rng(seed)
    k_arr = np.asarray(list(ks)[:n_steps], dtype=np.float64)
    x_base = np.log2(k_arr + 1.0)
    n_steps = len(x_base)

    x = np.tile(x_base, n_facts)
    cluster = np.repeat(np.arange(n_facts), n_steps)
    z_crit = float(norm.ppf(1.0 - alpha / 2))

    def draw(slope: float) -> np.ndarray:
        # Fact offsets cancel in the difference; only the two noise draws remain.
        noise = rng.normal(0.0, noise_sd, size=n_facts * n_steps) - rng.normal(
            0.0, noise_sd, size=n_facts * n_steps
        )
        fact_slope_noise = np.repeat(
            rng.normal(0.0, fact_sd, size=n_facts), n_steps
        )
        return slope * x + fact_slope_noise * x + noise

    reject = 0
    null_ses = []
    for _ in tqdm(range(n_sims), desc="power sims"):
        beta, se = _cluster_robust_slope_se(x, draw(effect_slope), cluster)
        if se > 0 and abs(beta / se) > z_crit:
            reject += 1
        _, null_se = _cluster_robust_slope_se(x, draw(0.0), cluster)
        null_ses.append(null_se)

    mean_se = float(np.mean(null_ses))
    # Effect detectable with 80% power at this alpha.
    mdd = float((z_crit + norm.ppf(0.80)) * mean_se)
    return {
        "estimator": "paired_learned_minus_control_slope",
        "n_facts": n_facts,
        "n_steps": n_steps,
        "n_sims": n_sims,
        "assumed_effect_slope": effect_slope,
        "fact_sd": fact_sd,
        "noise_sd": noise_sd,
        "estimated_power": reject / n_sims,
        "mean_slope_se": mean_se,
        "minimum_detectable_slope": mdd,
        "alpha": alpha,
    }


def estimate_curve_variance(
    curves: Any, *, treatment: str = "learned", reference: str = "control"
) -> dict[str, float]:
    """Fact-level and residual SD of the paired difference, from pilot curves.

    Feeding these into `simulate_power` replaces the hardcoded guesses with the
    dispersion the experiment actually produces.
    """
    from analysis.hierarchical import paired_curve_difference

    paired = paired_curve_difference(
        curves, treatment=treatment, reference=reference, checkpoint_role=None
    )
    if paired.empty:
        return {"fact_sd": 0.5, "noise_sd": 0.3, "n_facts": 0, "source": "default"}

    fact_means = paired.groupby("fact_id")["delta_gain"].mean()
    resid = paired["delta_gain"] - paired["fact_id"].map(fact_means)
    return {
        "fact_sd": float(fact_means.std(ddof=1)) if len(fact_means) > 1 else 0.0,
        "noise_sd": float(resid.std(ddof=1)) if len(resid) > 1 else 0.0,
        "n_facts": int(paired["fact_id"].nunique()),
        "source": "pilot_curves",
    }


def margin_feasibility(
    *,
    per_fact_sd: float,
    n_per_condition: int,
    margin: float,
    alpha: float = 0.05,
    label: str = "",
) -> dict[str, Any]:
    """Can a two-one-sided-test interval fit inside `margin` at this sample size?

    Equivalence requires the whole interval inside +/- margin, so the CI
    half-width must be smaller than the margin even when the point estimate is
    exactly zero. Learned and control are disjoint fact sets, hence the sqrt(2)
    from differencing two independent group means.
    """
    z = float(norm.ppf(1.0 - alpha / 2))
    se = per_fact_sd * np.sqrt(2.0 / max(n_per_condition, 1))
    half_width = z * se
    needed = int(np.ceil(2.0 * (z * per_fact_sd / margin) ** 2)) if margin > 0 else -1
    return {
        "quantity": label,
        "per_fact_sd": float(per_fact_sd),
        "n_per_condition": int(n_per_condition),
        "margin": float(margin),
        "alpha": alpha,
        "ci_half_width": float(half_width),
        "feasible": bool(half_width < margin),
        "n_per_condition_needed": needed,
        "shortfall_factor": float(half_width / margin) if margin > 0 else float("inf"),
    }


def write_power_report(path: str, report: dict[str, Any]) -> None:
    write_json(path, report)
