"""Equivalence margin checks for workshop behavioral criteria."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BehavioralMargins:
    log_odds: float = 0.15
    js_distance: float = 0.02


def simultaneous_alpha(alpha: float, n_tests: int, method: str = "none") -> float:
    n = max(1, int(n_tests))
    if method == "none" or n == 1:
        return alpha
    if method == "bonferroni":
        return alpha / n
    if method == "sidak":
        return 1.0 - (1.0 - alpha) ** (1.0 / n)
    raise ValueError(f"Unknown simultaneous correction: {method}")


def mean_ci(
    values: Sequence[float],
    alpha: float = 0.05,
    *,
    n_boot: int = 0,
    seed: int = 0,
) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        draws = np.array(
            [rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(n_boot)]
        )
        return (
            mean,
            float(np.quantile(draws, alpha / 2)),
            float(np.quantile(draws, 1 - alpha / 2)),
        )
    se = float(arr.std(ddof=1) / np.sqrt(arr.size))
    from scipy.stats import norm

    z = float(norm.ppf(1.0 - alpha / 2))
    return mean, mean - z * se, mean + z * se


def delta_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    arr_a = np.asarray(list(a), dtype=np.float64)
    arr_b = np.asarray(list(b), dtype=np.float64)
    if arr_a.size == 0 or arr_b.size == 0:
        return 0.0, 0.0, 0.0
    point = float(arr_a.mean() - arr_b.mean())
    if n_boot <= 0:
        return point, point, point
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        draws[i] = (
            rng.choice(arr_a, size=arr_a.size, replace=True).mean()
            - rng.choice(arr_b, size=arr_b.size, replace=True).mean()
        )
    return (
        point,
        float(np.quantile(draws, alpha / 2)),
        float(np.quantile(draws, 1 - alpha / 2)),
    )


def behavioral_equivalent(
    *,
    delta_log_odds_ci: tuple[float, float],
    mean_js_ci: tuple[float, float],
    margins: BehavioralMargins,
) -> bool:
    """TOST on target log-odds and JS upper-bound criterion."""
    lo_lo, lo_hi = delta_log_odds_ci
    js_upper = mean_js_ci[1]
    return (
        -margins.log_odds <= lo_lo
        and lo_hi <= margins.log_odds
        and js_upper < margins.js_distance
    )
