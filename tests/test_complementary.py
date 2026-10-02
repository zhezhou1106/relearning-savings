"""Complementary assignment tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facts.assign import assign_complementary_histories
from facts.generate import FactGenConfig, generate_candidates, select_balanced_panel


def test_three_seeds_complementary_balance():
    cfg = FactGenConfig(n_entities_per_pair=4, candidate_buffer=1.0, seed=3)
    panel = select_balanced_panel(generate_candidates(cfg), n_entities_per_pair=4)
    runs = assign_complementary_histories(panel, n_seeds=3, seed=3)
    assert set(runs) == {(s, c) for s in range(3) for c in (0, 1)}
    for s in range(3):
        for fid in [f.fact_id for f in panel]:
            assert {runs[(s, 0)][fid]["history"], runs[(s, 1)][fid]["history"]} == {
                "learned",
                "control",
            }
