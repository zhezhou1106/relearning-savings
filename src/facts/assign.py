"""Split facts and assign complementary learned/control histories per seed."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from tqdm import tqdm

from facts.generate import ANSWER_BANK, REGISTRY_RELATION, generate_entity_pool
from schema import COMPLEMENTS, RUNS, THREE_ARM_HISTORIES, Fact, History, Split


def partition_splits(
    facts: list[Fact],
    *,
    development: int = 384,
    confirmatory: int = 192,
    seed: int = 42,
) -> dict[Split, list[str]]:
    """Balanced partition by relation and answer identity (8+4 per cell)."""
    total = development + confirmatory
    if len(facts) != total:
        raise ValueError(f"Expected {total} facts, got {len(facts)}")

    rng = np.random.default_rng(seed)
    by_cell: dict[tuple[str, int], list[Fact]] = defaultdict(list)
    for fact in facts:
        by_cell[(fact.relation, fact.answer_id)].append(fact)

    buckets: dict[Split, list[str]] = {"development": [], "confirmatory": []}
    # 12 entities/cell → 8 development + 4 confirmatory
    per_cell = {
        "development": development // (6 * 8),
        "confirmatory": confirmatory // (6 * 8),
    }

    for cell_facts in by_cell.values():
        order = list(cell_facts)
        rng.shuffle(order)
        i = 0
        for split, n in per_cell.items():
            buckets[split].extend(f.fact_id for f in order[i : i + n])
            i += n

    return buckets


def exposure_entity_map(facts: list[Fact], *, seed: int) -> dict[str, str]:
    """One dedicated answer-exposure entity per fact, disjoint from the panel."""
    panel_entities = {f.entity for f in facts}
    pool = generate_entity_pool(
        len(facts), seed=seed, exclude_entities=panel_entities
    )
    return dict(zip(sorted(f.fact_id for f in facts), pool, strict=True))


def _wrong_answer_for(fact: Fact) -> tuple[str, int]:
    """Fixed cyclic shift within the relation's answer bank (never the gold)."""
    answers = list(ANSWER_BANK[fact.relation])
    # Deterministic shift of +1 (mod bank size) — uniform over the bank and
    # identical across seeds/runs so W means the same thing everywhere.
    wrong_id = (fact.answer_id + 1) % len(answers)
    wrong = answers[wrong_id]
    if wrong == fact.answer:
        raise AssertionError(f"wrong-answer collided with gold for {fact.fact_id}")
    return wrong, wrong_id


def _payload_for(
    fact: Fact,
    history: History,
    *,
    exposure_entities: dict[str, str],
    registry_answers: list[str],
) -> dict[str, Any]:
    registry_answer_id = fact.answer_id % len(registry_answers)
    payload: dict[str, Any] = {
        "fact_id": fact.fact_id,
        "history": history,
        "relation": fact.relation,
        "entity": fact.entity,
        "target_answer": fact.answer,
        "target_answer_id": fact.answer_id,
        "registry_answer": registry_answers[registry_answer_id],
        "registry_answer_id": registry_answer_id,
    }
    if history == "learned":
        payload["train_relation"] = fact.relation
        payload["train_answer"] = fact.answer
        payload["train_answer_id"] = fact.answer_id
    elif history == "wrong":
        wrong, wrong_id = _wrong_answer_for(fact)
        payload["train_relation"] = fact.relation
        payload["train_answer"] = wrong
        payload["train_answer_id"] = wrong_id
        payload["wrong_answer"] = wrong
        payload["wrong_answer_id"] = wrong_id
    else:
        payload["train_relation"] = REGISTRY_RELATION
        payload["train_answer"] = registry_answers[registry_answer_id]
        payload["train_answer_id"] = registry_answer_id
        payload["answer_exposure_entity"] = exposure_entities[fact.fact_id]
    return payload


def assign_complementary_histories(
    facts: list[Fact],
    *,
    n_seeds: int = 3,
    seed: int = 42,
) -> dict[tuple[int, int], dict[str, dict[str, Any]]]:
    """Assign learned/control with within-seed complementary swap.

    Returns maps keyed by (seed, complementary) where complementary in {0,1}.
    Within each seed, every fact is learned in one complementary run and control
    in the other.
    """
    if n_seeds < 1:
        raise ValueError("n_seeds must be >= 1")

    rng = np.random.default_rng(seed)
    facts_by_id = {f.fact_id: f for f in facts}
    fact_ids = sorted(facts_by_id)
    panel_entities = {f.entity for f in facts}
    exposure_entities = exposure_entity_map(facts, seed=seed + 900)
    registry_answers = list(ANSWER_BANK[REGISTRY_RELATION])

    run_maps: dict[tuple[int, int], dict[str, dict[str, Any]]] = {
        (s, c): {} for s in range(n_seeds) for c in COMPLEMENTS
    }

    for opt_seed in tqdm(range(n_seeds), desc="assign seeds"):
        # Fresh shuffle per seed so history assignment is not identical across seeds.
        order = list(fact_ids)
        rng.shuffle(order)
        for idx, fact_id in enumerate(order):
            fact = facts_by_id[fact_id]
            # Balanced: half start as learned on c0.
            c0_history: History = "learned" if idx % 2 == 0 else "control"
            c1_history: History = "control" if c0_history == "learned" else "learned"
            run_maps[(opt_seed, 0)][fact_id] = _payload_for(
                fact, c0_history, exposure_entities=exposure_entities, registry_answers=registry_answers
            )
            run_maps[(opt_seed, 1)][fact_id] = _payload_for(
                fact, c1_history, exposure_entities=exposure_entities, registry_answers=registry_answers
            )

    _validate_complementary(run_maps, fact_ids, n_seeds)
    _validate_exposure_entities(run_maps, panel_entities)
    return run_maps


def assign_three_arm_histories(
    facts: list[Fact],
    *,
    n_seeds: int = 3,
    seed: int = 42,
) -> dict[tuple[int, int], dict[str, dict[str, Any]]]:
    """3-cycle Latin square: learned / control / wrong across RUNS=(0,1,2).

    Partition the panel into 3 groups of equal size stratified by
    (relation, answer_id) cell. Run r assigns group g the arm
    THREE_ARM_HISTORIES[(g + r) % 3]. Every fact visits all three arms exactly
    once across the three runs; each run holds exactly n/3 facts per arm.
    """
    if n_seeds < 1:
        raise ValueError("n_seeds must be >= 1")
    if len(facts) % 3 != 0:
        raise ValueError(f"Panel size {len(facts)} is not divisible by 3")

    rng = np.random.default_rng(seed + 17)
    facts_by_id = {f.fact_id: f for f in facts}
    panel_entities = {f.entity for f in facts}
    exposure_entities = exposure_entity_map(facts, seed=seed + 901)
    registry_answers = list(ANSWER_BANK[REGISTRY_RELATION])
    arms = list(THREE_ARM_HISTORIES)

    # Stratify by cell into 3 groups of equal size.
    by_cell: dict[tuple[str, int], list[str]] = defaultdict(list)
    for fact in facts:
        by_cell[(fact.relation, fact.answer_id)].append(fact.fact_id)
    group_of: dict[str, int] = {}
    for cell_ids in by_cell.values():
        order = list(cell_ids)
        rng.shuffle(order)
        if len(order) % 3 != 0:
            raise ValueError(
                f"Cell size {len(order)} is not divisible by 3; "
                "need a multiple of 3 entities per (relation, answer) cell"
            )
        per = len(order) // 3
        for g in range(3):
            for fid in order[g * per : (g + 1) * per]:
                group_of[fid] = g

    run_maps: dict[tuple[int, int], dict[str, dict[str, Any]]] = {
        (s, r): {} for s in range(n_seeds) for r in RUNS
    }
    for opt_seed in tqdm(range(n_seeds), desc="assign 3-arm seeds"):
        # Fresh within-seed shuffle of cell membership is already fixed above;
        # re-shuffle groups across seeds by rotating the arm offset.
        seed_offset = int(rng.integers(0, 3))
        for fact_id, g in group_of.items():
            fact = facts_by_id[fact_id]
            for r in RUNS:
                history: History = arms[(g + r + seed_offset) % 3]  # type: ignore[assignment]
                run_maps[(opt_seed, r)][fact_id] = _payload_for(
                    fact,
                    history,
                    exposure_entities=exposure_entities,
                    registry_answers=registry_answers,
                )

    _validate_three_arm(run_maps, sorted(facts_by_id), n_seeds)
    _validate_exposure_entities(run_maps, panel_entities)
    return run_maps


def exposure_audit(
    facts: list[Fact],
    run_maps: dict[tuple[int, int], dict[str, dict[str, Any]]],
    phase_a_templates: dict[str, list[str]],
) -> dict[str, Any]:
    """Descriptive exposure counts for appendix reporting (no model runs)."""
    from data.datasets import build_phase_a_examples

    audit: dict[str, Any] = {"seeds": {}}
    for (opt_seed, comp), assignments in sorted(run_maps.items()):
        examples = build_phase_a_examples(facts, assignments, phase_a_templates)
        entity_counts: Counter[str] = Counter()
        answer_counts: Counter[str] = Counter()
        relation_counts: Counter[str] = Counter()
        history_counts: Counter[str] = Counter()
        template_counts: Counter[str] = Counter()
        for ex in examples:
            history_counts[str(ex["history"])] += 1
            relation_counts[str(ex["relation"])] += 1
            template_counts[f"{ex['relation']}:{ex['template_idx']}"] += 1
            # Entity/answer strings appear in rendered text; use assignment metadata.
            fid = str(ex["fact_id"])
            asg = assignments[fid]
            entity_counts[str(asg["entity"])] += 1
            if asg["history"] == "learned":
                answer_counts[str(asg["target_answer"])] += 1
            elif "answer_exposure_entity" in asg and ex["history"] == "control_answer_exposure":
                entity_counts[str(asg["answer_exposure_entity"])] += 1
                answer_counts[str(asg["target_answer"])] += 1
            else:
                answer_counts[str(asg["train_answer"])] += 1

        key = f"seed_{opt_seed}_c{comp}"
        audit["seeds"][key] = {
            "n_examples": len(examples),
            "n_updates_at_batch_64": len(examples) / 64.0,
            "history_counts": dict(history_counts),
            "relation_counts": dict(relation_counts),
            "n_unique_entities": len(entity_counts),
            "n_unique_answers": len(answer_counts),
            "n_template_slots": len(template_counts),
            "mean_entity_exposures": float(np.mean(list(entity_counts.values())))
            if entity_counts
            else 0.0,
            "mean_answer_exposures": float(np.mean(list(answer_counts.values())))
            if answer_counts
            else 0.0,
        }
    # Pairwise / three-way balance summary within each seed.
    balance = {}
    seeds = sorted({k[0] for k in run_maps})
    for s in seeds:
        runs = sorted(c for (ss, c) in run_maps if ss == s)
        counts = {
            f"c{c}": audit["seeds"][f"seed_{s}_c{c}"]["n_examples"] for c in runs
        }
        balance[f"seed_{s}"] = {
            "example_count_match": len(set(counts.values())) == 1,
            "n_examples": next(iter(counts.values())) if counts else 0,
            "n_runs": len(runs),
            "per_run": counts,
        }
    audit["complementary_balance"] = balance
    return audit


def _validate_complementary(
    run_maps: dict[tuple[int, int], dict[str, dict[str, Any]]],
    fact_ids: list[str],
    n_seeds: int,
) -> None:
    for s in range(n_seeds):
        for fact_id in fact_ids:
            h0 = run_maps[(s, 0)][fact_id]["history"]
            h1 = run_maps[(s, 1)][fact_id]["history"]
            if {h0, h1} != {"learned", "control"}:
                raise AssertionError(
                    f"seed {s} fact {fact_id}: complementary histories are {h0!r}, {h1!r}"
                )
        for c in COMPLEMENTS:
            counts = Counter(p["history"] for p in run_maps[(s, c)].values())
            if counts["learned"] != counts["control"]:
                raise AssertionError(
                    f"seed {s} c{c}: unbalanced histories {dict(counts)}"
                )


def _validate_three_arm(
    run_maps: dict[tuple[int, int], dict[str, dict[str, Any]]],
    fact_ids: list[str],
    n_seeds: int,
) -> None:
    expected = set(THREE_ARM_HISTORIES)
    n_per_arm = len(fact_ids) // 3
    for s in range(n_seeds):
        for fact_id in fact_ids:
            visited = {run_maps[(s, r)][fact_id]["history"] for r in RUNS}
            if visited != expected:
                raise AssertionError(
                    f"seed {s} fact {fact_id}: visited {visited}, expected {expected}"
                )
        for r in RUNS:
            counts = Counter(p["history"] for p in run_maps[(s, r)].values())
            for arm in THREE_ARM_HISTORIES:
                if counts[arm] != n_per_arm:
                    raise AssertionError(
                        f"seed {s} c{r}: arm {arm} has {counts[arm]}, expected {n_per_arm}"
                    )


def _validate_exposure_entities(
    run_maps: dict[tuple[int, int], dict[str, dict[str, Any]]],
    panel_entities: set[str],
) -> None:
    for key, payloads in run_maps.items():
        seen: dict[str, str] = {}
        for fact_id, payload in payloads.items():
            entity = payload.get("answer_exposure_entity")
            if entity is None:
                continue
            if entity in panel_entities:
                raise AssertionError(
                    f"{key}: answer-exposure entity {entity!r} for {fact_id} "
                    "is a panel entity"
                )
            if entity in seen:
                raise AssertionError(
                    f"{key}: exposure entity {entity!r} reused by {fact_id} "
                    f"and {seen[entity]}"
                )
            seen[entity] = fact_id
