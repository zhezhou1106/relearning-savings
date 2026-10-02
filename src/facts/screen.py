"""Base-model screening and answer matching for synthetic facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from tqdm import tqdm

from facts.generate import (
    ANSWER_BANK,
    FactGenConfig,
    contradictory_facts,
    duplicate_entity_facts,
    generate_candidates,
    select_balanced_panel,
)
from schema import Fact


@dataclass
class ScreenResult:
    kept: list[Fact]
    rejected: list[dict[str, Any]]
    summary: dict[str, Any]


def _answer_token_lengths(tokenizer: Any, answers: list[str]) -> dict[str, int]:
    # Space-prefixed to match how every scorer encodes answers.
    return {a: len(tokenizer.encode(" " + a, add_special_tokens=False)) for a in answers}


def _target_favored(
    model: Any,
    tokenizer: Any,
    fact: Fact,
    answers: list[str],
    device: str,
) -> tuple[bool, list[float]]:
    """Return whether base model ranks target answer #1 among relation answers."""
    import torch

    prompt = (
        f"In the invented registry, the {fact.relation} of {fact.entity} is"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
        next_logits = out.logits[0, -1, :]

    scores: list[float] = []
    for answer in answers:
        token_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
        if not token_ids:
            scores.append(float("-inf"))
            continue
        # Score by first-token logprob as a cheap prior screen.
        scores.append(float(next_logits[token_ids[0]].item()))
    ranked = int(np.argmax(scores))
    return ranked == fact.answer_id, scores


def screen_facts(
    facts: list[Fact],
    *,
    model: Any | None = None,
    tokenizer: Any | None = None,
    device: str = "cpu",
    dry_run: bool = False,
    max_reject_frac: float = 0.4,
    n_entities_per_pair: int = 12,
) -> ScreenResult:
    """Screen candidates; dry_run keeps a balanced panel without a model."""
    rejected: list[dict[str, Any]] = []
    kept: list[Fact] = []

    if dry_run or model is None or tokenizer is None:
        ordered = sorted(facts, key=lambda f: f.fact_id)
        kept = select_balanced_panel(ordered, n_entities_per_pair=n_entities_per_pair)
        summary = {
            "mode": "dry_run",
            "n_candidates": len(facts),
            "n_kept": len(kept),
            "n_rejected": 0,
        }
        return ScreenResult(kept=kept, rejected=rejected, summary=summary)

    by_relation: dict[str, list[Fact]] = {}
    for fact in facts:
        by_relation.setdefault(fact.relation, []).append(fact)

    for relation, rel_facts in by_relation.items():
        answers = list(ANSWER_BANK[relation])
        lengths = _answer_token_lengths(tokenizer, answers)
        modal = max(set(lengths.values()), key=list(lengths.values()).count)
        for fact in tqdm(rel_facts, desc=f"screen {relation}", leave=False):
            if abs(lengths[fact.answer] - modal) > 1:
                rejected.append(
                    {"fact_id": fact.fact_id, "reason": "token_length_mismatch"}
                )
                continue
            favored, scores = _target_favored(model, tokenizer, fact, answers, device)
            if favored:
                rejected.append(
                    {
                        "fact_id": fact.fact_id,
                        "reason": "base_favors_target",
                        "scores": scores,
                    }
                )
                continue
            kept.append(fact)

    try:
        kept = select_balanced_panel(kept, n_entities_per_pair=n_entities_per_pair)
    except ValueError:
        seen_entities = {f.entity for f in facts} | {f.entity for f in kept}
        extra = generate_candidates(
            FactGenConfig(
                n_entities_per_pair=n_entities_per_pair,
                candidate_buffer=2.0,
                seed=99,
            ),
            exclude_entities=seen_entities,
        )
        pool = kept + [f for f in extra if f.fact_id not in {k.fact_id for k in kept}]
        kept = select_balanced_panel(pool, n_entities_per_pair=n_entities_per_pair)

    reject_frac = len(rejected) / max(len(facts), 1)
    duplicates = duplicate_entity_facts(kept)
    contradictions = contradictory_facts(kept)
    summary = {
        "mode": "model_screen",
        "n_candidates": len(facts),
        "n_kept": len(kept),
        "n_rejected": len(rejected),
        "reject_frac": reject_frac,
        "max_reject_frac": max_reject_frac,
        "warn_high_reject": reject_frac > max_reject_frac,
        "n_duplicate_entities": len(duplicates),
        "n_contradictory_bindings": len(contradictions),
    }
    if contradictions:
        raise ValueError(
            f"Panel contains {len(contradictions)} (entity, relation) pairs with "
            f"more than one answer, e.g. {sorted(contradictions)[:3]}. These teach "
            "the model contradictory targets; regenerate the panel."
        )
    return ScreenResult(kept=kept, rejected=rejected, summary=summary)
