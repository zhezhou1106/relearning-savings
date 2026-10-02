"""Equivalence margin and checkpoint nomination tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checkpoints.equivalence import BehavioralMargins, behavioral_equivalent
from checkpoints.select import select_checkpoints


def test_behavioral_margins_inside():
    assert behavioral_equivalent(
        delta_log_odds_ci=(-0.05, 0.05),
        mean_js_ci=(0.0, 0.01),
        margins=BehavioralMargins(),
    )


def test_behavioral_margins_outside_log_odds():
    assert not behavioral_equivalent(
        delta_log_odds_ci=(-0.2, 0.2),
        mean_js_ci=(0.0, 0.01),
        margins=BehavioralMargins(),
    )


def test_behavioral_margins_outside_js():
    assert not behavioral_equivalent(
        delta_log_odds_ci=(-0.05, 0.05),
        mean_js_ci=(0.0, 0.05),
        margins=BehavioralMargins(),
    )


def test_select_earliest_equivalence():
    timeline = [
        {"stream_steps": 0, "behavioral_equivalent": False},
        {"stream_steps": 1000, "behavioral_equivalent": False},
        {"stream_steps": 4000, "behavioral_equivalent": True},
        {"stream_steps": 8000, "behavioral_equivalent": True},
    ]
    sel = select_checkpoints(seed=0, timeline=timeline)
    assert sel.equivalence_steps == 4000
