"""Equivalence decisions must rest on real intervals, not point estimates."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from checkpoints.equivalence import (  # noqa: E402
    BehavioralMargins,
    behavioral_equivalent,
    delta_ci,
    simultaneous_alpha,
)
from checkpoints.select import select_checkpoints  # noqa: E402
from eval.behavioral import aggregate_paired_fact_gaps  # noqa: E402
from schema import BehavioralScores  # noqa: E402


def make_score(
    fact_id: str, history: str, *, log_odds: float, em: float, answer_id: int = 0
) -> BehavioralScores:
    probs = np.full(8, 0.05)
    probs[answer_id] = 1.0 - 0.05 * 7
    return BehavioralScores(
        fact_id=fact_id,
        history=history,  # type: ignore[arg-type]
        exact_match=em,
        target_log_odds=log_odds,
        target_rank=1,
        answer_probs=tuple(float(x) for x in probs),
    )


def test_simultaneous_alpha_none_default() -> None:
    assert simultaneous_alpha(0.05, 2, method="none") == 0.05
    assert simultaneous_alpha(0.05, 4, method="bonferroni") == pytest.approx(0.0125)


def test_delta_ci_is_not_degenerate() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 2.0, size=64)
    b = rng.normal(0.0, 2.0, size=64)
    point, lo, hi = delta_ci(a, b, n_boot=500, seed=1)
    assert hi > lo
    assert lo < point < hi


def test_wide_interval_blocks_equivalence() -> None:
    margins = BehavioralMargins()
    assert not behavioral_equivalent(
        delta_log_odds_ci=(-0.7, 0.7),
        mean_js_ci=(0.0, 0.001),
        margins=margins,
    )
    assert behavioral_equivalent(
        delta_log_odds_ci=(-0.05, 0.05),
        mean_js_ci=(0.0, 0.001),
        margins=margins,
    )


def test_paired_gaps_by_fact() -> None:
    learned = {
        f"f{i}": make_score(f"f{i}", "learned", log_odds=1.0, em=1.0) for i in range(8)
    }
    control = {
        f"f{i}": make_score(f"f{i}", "control", log_odds=0.0, em=0.0) for i in range(8)
    }
    gaps = aggregate_paired_fact_gaps(learned, control, n_boot=200, seed=0)
    assert gaps["delta_log_odds"] == pytest.approx(1.0)
    assert gaps["n_facts"] == 8


def test_select_from_cis() -> None:
    timeline = [
        {
            "stream_steps": 500,
            "delta_log_odds_ci": [-0.5, 0.5],
            "mean_js_ci": [0.0, 0.01],
        },
        {
            "stream_steps": 2000,
            "delta_log_odds_ci": [-0.05, 0.05],
            "mean_js_ci": [0.0, 0.01],
        },
    ]
    sel = select_checkpoints(seed=1, timeline=timeline)
    assert sel.equivalence_steps == 2000
