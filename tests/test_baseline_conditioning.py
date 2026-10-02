"""Tests for baseline-conditioned savings, threshold savings, and 3-arm assignment."""

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
    choose_threshold_on_development,
    fact_level_savings,
    pooled_gain_residual_contrast,
    savings_baseline_regression,
    threshold_savings,
    updates_to_threshold,
)
from facts.assign import assign_three_arm_histories  # noqa: E402
from facts.generate import FactGenConfig, generate_candidates, select_balanced_panel  # noqa: E402


KS = [0, 1, 2, 4, 8, 16, 32, 64]


def _make_curves(
    *,
    learned_advantage: float,
    baseline_effect: float = 0.0,
    n_facts: int = 60,
    noise_sd: float = 0.02,
    seed: int = 0,
) -> pd.DataFrame:
    """Plant curves where s_i = advantage + baseline_effect * d_i."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_facts):
        fact_id = f"fact_{i:03d}"
        d_i = rng.normal(0.0, 0.5)
        base_c = rng.normal(-1.0, 0.2)
        base_l = base_c + d_i
        s_i = learned_advantage + baseline_effect * d_i
        for history, base in (("learned", base_l), ("control", base_c)):
            for k in KS:
                extra = (s_i if history == "learned" else 0.0) if k > 0 else 0.0
                lo = base + 0.05 * np.log2(k + 1.0) + extra + rng.normal(0.0, noise_sd)
                rows.append(
                    {
                        "fact_id": fact_id,
                        "history": history,
                        "k": k,
                        "target_log_odds": lo,
                        "baseline_log_odds": base,
                        "gain": lo - base,
                        "target_prob": float(1.0 / (1.0 + np.exp(-lo))),
                        "checkpoint_role": "forgetting_sweep",
                        "stream_steps": 100,
                        "seed": 0,
                    }
                )
    return pd.DataFrame(rows)


def test_savings_baseline_regression_recovers_intercept() -> None:
    df = _make_curves(learned_advantage=0.25, baseline_effect=-0.4, noise_sd=0.01)
    reg = savings_baseline_regression(
        df, checkpoint_role="forgetting_sweep", primary_ks=PRIMARY_KS, n_boot=400, seed=1
    )
    assert reg["intercept"] == pytest.approx(0.25, abs=0.05)
    assert reg["slope"] == pytest.approx(-0.4, abs=0.1)
    assert reg["primary_pass"] is True


def test_savings_baseline_regression_external_d() -> None:
    df = _make_curves(learned_advantage=0.3, baseline_effect=0.0, noise_sd=0.01)
    fact = fact_level_savings(df, checkpoint_role="forgetting_sweep")
    # External d_i uncorrelated with relearn-k0 d — intercept still recovers.
    rng = np.random.default_rng(0)
    external = {fid: float(rng.normal(0, 0.3)) for fid in fact["fact_id"]}
    reg = savings_baseline_regression(
        df,
        checkpoint_role="forgetting_sweep",
        baseline_deltas=external,
        d_source="behavioral_eval",
        n_boot=300,
        seed=2,
    )
    assert reg["intercept"] == pytest.approx(0.3, abs=0.08)


def test_pooled_gain_residual_positive() -> None:
    df = _make_curves(learned_advantage=0.5, baseline_effect=-0.3, noise_sd=0.01)
    # Duplicate across stream steps so the pooled fit has enough mass.
    parts = []
    for step in (50, 100, 150):
        sub = df.copy()
        sub["stream_steps"] = step
        parts.append(sub)
    pooled = pd.concat(parts, ignore_index=True)
    out = pooled_gain_residual_contrast(
        pooled,
        primary_ks=PRIMARY_KS,
        stream_steps=100,
        n_boot=300,
        seed=3,
        frac=0.5,
    )
    assert out["residual_contrast"] > 0.2
    assert out["primary_pass"] is True


def test_updates_to_threshold_interpolation() -> None:
    rows = []
    for k, p in ((0, 0.1), (1, 0.2), (2, 0.4), (4, 0.8), (8, 0.95)):
        rows.append(
            {
                "fact_id": "f0",
                "history": "learned",
                "k": k,
                "target_prob": p,
            }
        )
    df = pd.DataFrame(rows)
    utt = updates_to_threshold(df, threshold=0.5)
    assert len(utt) == 1
    assert not bool(utt.iloc[0]["censored"])
    # Crossing is between k=2 (0.4) and k=4 (0.8).
    assert 2.0 < float(utt.iloc[0]["updates"]) < 4.0


def test_threshold_savings_positive() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        fid = f"f{i:02d}"
        for history, speed in (("learned", 1.0), ("control", 0.4)):
            for k in KS:
                # Learned climbs faster.
                p = 1.0 / (1.0 + np.exp(-(speed * np.log2(k + 1.0) - 1.5)))
                p = float(np.clip(p + rng.normal(0, 0.01), 0.01, 0.99))
                rows.append(
                    {
                        "fact_id": fid,
                        "history": history,
                        "k": k,
                        "target_prob": p,
                    }
                )
    df = pd.DataFrame(rows)
    thr_info = choose_threshold_on_development(df, min_reach_rate=0.5)
    result = threshold_savings(df, threshold=float(thr_info["threshold"]), n_boot=200, seed=1)
    assert result["updates_saved"] > 0
    assert result["primary_pass"] is True


def test_three_arm_latin_square() -> None:
    # 6×8×6 = 288 facts (divisible by 3; 6 per cell → 2 per group).
    cfg = FactGenConfig(n_entities_per_pair=6, candidate_buffer=1.0, seed=5)
    panel = select_balanced_panel(generate_candidates(cfg), n_entities_per_pair=6)
    runs = assign_three_arm_histories(panel, n_seeds=2, seed=5)
    assert set(runs) == {(s, r) for s in range(2) for r in (0, 1, 2)}
    for s in range(2):
        for fid in [f.fact_id for f in panel]:
            visited = {runs[(s, r)][fid]["history"] for r in (0, 1, 2)}
            assert visited == {"learned", "control", "wrong"}
        for r in (0, 1, 2):
            counts = {}
            for p in runs[(s, r)].values():
                counts[p["history"]] = counts.get(p["history"], 0) + 1
            assert counts["learned"] == counts["control"] == counts["wrong"] == 96
            # Wrong answers are never the gold.
            for p in runs[(s, r)].values():
                if p["history"] == "wrong":
                    assert p["train_answer"] != p["target_answer"]
                    assert p["train_relation"] == p["relation"]
