"""Behavioral metric helpers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval.behavioral import aggregate_paired_fact_gaps, js_distance
from schema import BehavioralScores


def test_js_distance_identical():
    p = (0.125,) * 8
    assert js_distance(p, p) == 0.0


def test_aggregate_paired_gaps():
    learned = {}
    control = {}
    for i in range(4):
        fid = f"f{i}"
        learned[fid] = BehavioralScores(
            fact_id=fid,
            history="learned",
            exact_match=1.0,
            target_log_odds=1.0,
            target_rank=1,
            answer_probs=(0.5, 0.5 / 7, 0.5 / 7, 0.5 / 7, 0.5 / 7, 0.5 / 7, 0.5 / 7, 0.5 / 7),
        )
        control[fid] = BehavioralScores(
            fact_id=fid,
            history="control",
            exact_match=0.0,
            target_log_odds=0.0,
            target_rank=4,
            answer_probs=(0.125,) * 8,
        )
    gaps = aggregate_paired_fact_gaps(learned, control, n_boot=100, seed=0)
    assert gaps["delta_log_odds"] == 1.0
    assert gaps["delta_exact_match"] == 1.0
    assert gaps["mean_js"] >= 0.0
