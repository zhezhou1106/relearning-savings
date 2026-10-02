"""Primary early-update contrast recovers planted learned-minus-control savings."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis.hierarchical import (  # noqa: E402
    PRIMARY_KS,
    counterbalanced_bootstrap_ci,
    fit_primary_contrast,
)

KS = [0, 1, 2, 4, 8, 16, 32, 64]


def make_curves(
    *,
    learned_advantage: float,
    n_facts: int = 40,
    noise_sd: float = 0.02,
    seed: int = 0,
) -> pd.DataFrame:
    """Plant absolute LO curves where learned exceeds control by a fixed margin after k=0."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_facts):
        fact_id = f"fact_{i:03d}"
        base_l = rng.normal(0.0, 0.1)
        base_c = rng.normal(0.0, 0.1)
        for history, base in (("learned", base_l), ("control", base_c)):
            for k in KS:
                # After baseline, learned gains an extra advantage per early update.
                extra = learned_advantage if (history == "learned" and k > 0) else 0.0
                lo = base + 0.1 * np.log2(k + 1.0) + extra + rng.normal(0.0, noise_sd)
                rows.append(
                    {
                        "fact_id": fact_id,
                        "history": history,
                        "k": k,
                        "target_log_odds": lo,
                        "baseline_log_odds": base,
                        "gain": lo - base,
                        "checkpoint_role": "behavioral_equivalence",
                        "seed": 0,
                    }
                )
    return pd.DataFrame(rows)


def test_positive_advantage_yields_positive_early_contrast() -> None:
    df = make_curves(learned_advantage=0.5)
    fit = fit_primary_contrast(df, primary_ks=PRIMARY_KS)
    assert fit["early_contrast"] == pytest.approx(0.5, abs=0.05)


def test_zero_advantage_near_zero() -> None:
    df = make_curves(learned_advantage=0.0, noise_sd=0.0)
    fit = fit_primary_contrast(df, primary_ks=PRIMARY_KS)
    assert fit["early_contrast"] == pytest.approx(0.0, abs=1e-9)


def test_control_faster_fails_primary() -> None:
    df = make_curves(learned_advantage=-0.4)
    result = counterbalanced_bootstrap_ci(df, n_boot=200, seed=1, primary_ks=PRIMARY_KS)
    assert result["early_contrast"] < 0
    assert result["primary_pass"] is False


def test_bootstrap_ci_excludes_zero_when_strong() -> None:
    df = make_curves(learned_advantage=0.8, noise_sd=0.01)
    result = counterbalanced_bootstrap_ci(df, n_boot=300, seed=2, primary_ks=PRIMARY_KS)
    assert result["early_contrast_ci"][0] > 0
    assert result["primary_pass"] is True
