"""Dataset builders for Phase A/B and relearning."""

from __future__ import annotations

from typing import Any, Iterable

from facts.generate import ANSWER_BANK, REGISTRY_RELATION, RELATIONS, FactGenConfig, generate_candidates
from facts.templates import render
from schema import Fact


def build_phase_a_examples(
    facts: list[Fact],
    assignments: dict[str, dict[str, Any]],
    phase_a_templates: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Phase-A training texts, matched in example count across histories.

    Every fact contributes two blocks of `len(templates)` examples:

    - learned: (entity, relation, true answer) + (entity, registry_code filler)
    - control: (entity, registry_code) + (exposure entity, relation, true answer)

    Entity familiarity, answer familiarity, relation practice, and update count
    are matched; only the target binding differs.
    """
    examples: list[dict[str, Any]] = []
    for fact in facts:
        asg = assignments[fact.fact_id]
        history = asg["history"]
        train_relation = asg["train_relation"]
        train_answer = asg["train_answer"]
        templates = phase_a_templates[train_relation]
        proxy = Fact(
            fact_id=fact.fact_id,
            relation=train_relation,
            entity=fact.entity,
            answer=train_answer,
            answer_id=int(asg["train_answer_id"]),
        )
        for t_idx, template in enumerate(templates):
            examples.append(
                {
                    "text": render(template, proxy),
                    "fact_id": fact.fact_id,
                    "history": history,
                    "relation": train_relation,
                    "answer_id": proxy.answer_id,
                    "template_idx": t_idx,
                    "answer_text": train_answer,
                }
            )
        if history == "control":
            other = Fact(
                fact_id=f"exposure__{fact.fact_id}",
                relation=fact.relation,
                entity=asg["answer_exposure_entity"],
                answer=fact.answer,
                answer_id=fact.answer_id,
            )
            second_block = (other, fact.relation, "control_answer_exposure")
        else:
            registry = Fact(
                fact_id=f"registry__{fact.fact_id}",
                relation=REGISTRY_RELATION,
                entity=fact.entity,
                answer=asg["registry_answer"],
                answer_id=int(asg["registry_answer_id"]),
            )
            second_block = (registry, REGISTRY_RELATION, "learned_registry_filler")

        block_fact, block_relation, block_tag = second_block
        for t_idx, template in enumerate(phase_a_templates[block_relation]):
            examples.append(
                {
                    "text": render(template, block_fact),
                    "fact_id": fact.fact_id,
                    "history": block_tag,
                    "relation": block_relation,
                    "answer_id": block_fact.answer_id,
                    "template_idx": t_idx,
                    "answer_text": block_fact.answer,
                }
            )
    return examples


def _resolve_overwrite_facts(
    panel_facts: list[Fact],
    *,
    overwrite_scope: str,
    overwrite_relation: str | None,
) -> list[Fact]:
    if overwrite_scope == "none":
        return []
    if overwrite_scope == "full":
        return list(panel_facts)
    if overwrite_scope == "subset":
        relation = overwrite_relation or RELATIONS[0]
        subset = [f for f in panel_facts if f.relation == relation]
        if not subset:
            raise ValueError(
                f"overwrite_scope=subset but no panel facts for relation={relation!r}"
            )
        return subset
    raise ValueError(
        f"overwrite_scope must be none|subset|full, got {overwrite_scope!r}"
    )


def build_phase_b_stream(
    *,
    n_facts: int = 3000,
    seed: int = 123,
    phase_a_templates: dict[str, list[str]],
    exclude_entities: Iterable[str] | None = None,
    panel_facts: list[Fact] | None = None,
    overwrite_panel: bool = False,
    overwrite_scope: str | None = None,
    overwrite_relation: str | None = None,
    n_templates_per_fact: int = 2,
) -> list[dict[str, Any]]:
    """New entities, same relations; optional same-relation overwrite of panel entities.

    ``overwrite_scope`` controls wrong-answer supervision on Phase-A panel entities:

    - ``none``: pure new-entity stream (default baseline).
    - ``subset``: wrong answers for one primary relation (~1/6 of the panel).
    - ``full``: wrong answers for all 576 panel entities.

    Legacy ``overwrite_panel=True`` is equivalent to ``overwrite_scope='full'``.
    Both complementary histories receive the identical overwrite stream so the
    intervention is matched; only residual preference for the old binding can differ.
    """
    import numpy as np

    cfg = FactGenConfig(
        n_relations=6,
        n_answers_per_relation=8,
        n_entities_per_pair=max(1, n_facts // (6 * 8) + 1),
        candidate_buffer=1.0,
        seed=seed,
    )
    excluded = set(exclude_entities or ())
    candidates = generate_candidates(cfg, exclude_entities=excluded)[:n_facts]
    stream_entities = {f.entity for f in candidates}
    overlap = stream_entities & excluded
    if overlap:
        raise AssertionError(
            f"Phase-B stream re-used {len(overlap)} Phase-A entities, e.g. "
            f"{sorted(overlap)[:3]}"
        )
    examples: list[dict[str, Any]] = []
    n_tmpl = max(1, int(n_templates_per_fact))
    for fact in candidates:
        for t_idx, template in enumerate(phase_a_templates[fact.relation][:n_tmpl]):
            examples.append(
                {
                    "text": render(template, fact),
                    "fact_id": fact.fact_id,
                    "relation": fact.relation,
                    "answer_id": fact.answer_id,
                    "template_idx": t_idx,
                    "stream": True,
                    "answer_text": fact.answer,
                }
            )

    scope = overwrite_scope
    if scope is None:
        scope = "full" if overwrite_panel else "none"
    overwrite_facts = _resolve_overwrite_facts(
        panel_facts or [],
        overwrite_scope=scope,
        overwrite_relation=overwrite_relation,
    )
    if overwrite_facts:
        if not panel_facts:
            raise ValueError("panel_facts required when overwrite_scope != none")
        rng = np.random.default_rng(seed + 17)
        for fact in overwrite_facts:
            answers = list(ANSWER_BANK[fact.relation])
            # Deterministic wrong answer: shift by 1..7 so never the gold.
            shift = int(rng.integers(1, len(answers)))
            wrong_id = (fact.answer_id + shift) % len(answers)
            wrong = Fact(
                fact_id=f"overwrite__{fact.fact_id}",
                relation=fact.relation,
                entity=fact.entity,
                answer=answers[wrong_id],
                answer_id=wrong_id,
            )
            if wrong.answer == fact.answer:
                raise AssertionError(f"overwrite collided with gold for {fact.fact_id}")
            for t_idx, template in enumerate(phase_a_templates[fact.relation][:n_tmpl]):
                examples.append(
                    {
                        "text": render(template, wrong),
                        "fact_id": wrong.fact_id,
                        "relation": fact.relation,
                        "answer_id": wrong_id,
                        "template_idx": t_idx,
                        "stream": True,
                        "overwrite": True,
                        "source_fact_id": fact.fact_id,
                        "answer_text": wrong.answer,
                    }
                )

    rng = np.random.default_rng(seed + 91)
    order = rng.permutation(len(examples))
    return [examples[i] for i in order]


def build_relearn_examples(
    fact: Fact,
    relearn_templates: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "text": render(template, fact),
            "fact_id": fact.fact_id,
            "template_idx": i,
            "answer_text": fact.answer,
        }
        for i, template in enumerate(relearn_templates)
    ]


def relation_answer_list(relation: str) -> list[str]:
    return list(ANSWER_BANK[relation])


def all_primary_relations() -> tuple[str, ...]:
    return RELATIONS
