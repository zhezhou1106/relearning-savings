"""Tests for fact schema and generation balance (576-fact panel)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facts.generate import FactGenConfig, generate_candidates, select_balanced_panel
from schema import Fact


def test_fact_roundtrip():
    f = Fact("id", "element", "Aelara", "zoridium", 0)
    assert Fact.from_dict(f.to_dict()) == f


def test_generate_and_balance():
    cfg = FactGenConfig(n_entities_per_pair=12, candidate_buffer=1.25, seed=0)
    cands = generate_candidates(cfg)
    assert len(cands) >= 576
    panel = select_balanced_panel(cands, n_entities_per_pair=12)
    assert len(panel) == 576
    from collections import Counter

    counts = Counter((f.relation, f.answer_id) for f in panel)
    assert len(counts) == 48
    assert set(counts.values()) == {12}
