"""Complementary learned/control assignment and contamination guards."""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.datasets import build_phase_a_examples, build_phase_b_stream  # noqa: E402
from facts.assign import (  # noqa: E402
    assign_complementary_histories,
    exposure_entity_map,
    partition_splits,
)
from facts.generate import (  # noqa: E402
    FactGenConfig,
    contradictory_facts,
    duplicate_entity_facts,
    entity_namespace_size,
    generate_candidates,
    generate_entity_pool,
    select_balanced_panel,
)
from facts.templates import PHASE_A_TEMPLATES  # noqa: E402


@pytest.fixture(scope="module")
def panel():
    cfg = FactGenConfig(n_entities_per_pair=4, candidate_buffer=1.0, seed=7)
    return select_balanced_panel(generate_candidates(cfg), n_entities_per_pair=4)


def test_exposure_entities_are_disjoint_from_panel(panel) -> None:
    mapping = exposure_entity_map(panel, seed=900)
    panel_entities = {f.entity for f in panel}
    assert len(mapping) == len(panel)
    assert not (set(mapping.values()) & panel_entities)


def test_complementary_swap_within_seed(panel) -> None:
    runs = assign_complementary_histories(panel, n_seeds=2, seed=7)
    for fact in panel:
        h0 = runs[(0, 0)][fact.fact_id]["history"]
        h1 = runs[(0, 1)][fact.fact_id]["history"]
        assert {h0, h1} == {"learned", "control"}


def test_partition_development_confirmatory() -> None:
    cfg = FactGenConfig(n_entities_per_pair=12, candidate_buffer=1.0, seed=1)
    panel = select_balanced_panel(generate_candidates(cfg), n_entities_per_pair=12)
    splits = partition_splits(panel, development=384, confirmatory=192, seed=1)
    assert len(splits["development"]) == 384
    assert len(splits["confirmatory"]) == 192
    assert not (set(splits["development"]) & set(splits["confirmatory"]))


def test_phase_a_example_counts_match_across_histories(panel) -> None:
    runs = assign_complementary_histories(panel, n_seeds=1, seed=7)
    asg = runs[(0, 0)]
    examples = build_phase_a_examples(panel, asg, PHASE_A_TEMPLATES)
    per_history_totals: collections.Counter = collections.Counter()
    for ex in examples:
        per_history_totals[asg[ex["fact_id"]]["history"]] += 1
    fact_counts = collections.Counter(p["history"] for p in asg.values())
    per_fact = {h: per_history_totals[h] / fact_counts[h] for h in fact_counts}
    assert len(set(per_fact.values())) == 1, per_fact
    assert per_fact["learned"] == 2 * len(PHASE_A_TEMPLATES["element"])


def test_phase_a_never_trains_a_wrong_answer_on_a_panel_entity(panel) -> None:
    asg = assign_complementary_histories(panel, n_seeds=1, seed=7)[(0, 0)]
    examples = build_phase_a_examples(panel, asg, PHASE_A_TEMPLATES)
    truth = {(f.entity, f.relation): f.answer for f in panel}
    for ex in examples:
        for (entity, relation), answer in truth.items():
            if entity not in ex["text"] or relation != ex["relation"]:
                continue
            assert answer in ex["text"], (
                f"{asg[ex['fact_id']]['history']} example binds {entity}/{relation} "
                f"to something other than {answer}: {ex['text']!r}"
            )


def test_phase_b_stream_excludes_phase_a_entities(panel) -> None:
    asg = assign_complementary_histories(panel, n_seeds=1, seed=7)[(0, 0)]
    phase_a_entities = {f.entity for f in panel} | {
        p["answer_exposure_entity"]
        for p in asg.values()
        if p.get("answer_exposure_entity")
    }
    stream = build_phase_b_stream(
        n_facts=200,
        seed=1234,
        phase_a_templates=PHASE_A_TEMPLATES,
        exclude_entities=phase_a_entities,
    )
    stream_entities = {e["fact_id"].split("__")[1] for e in stream}
    assert not (stream_entities & phase_a_entities)


def test_entity_pool_refuses_to_exceed_the_namespace() -> None:
    with pytest.raises(ValueError, match="Cannot draw"):
        generate_entity_pool(entity_namespace_size() + 1, seed=0)


def test_panel_has_no_contradictory_bindings(panel) -> None:
    assert contradictory_facts(panel) == {}
    assert duplicate_entity_facts(panel) == {}
