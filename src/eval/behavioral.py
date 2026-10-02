"""Behavioral retention metrics (workshop: multi-token answer scoring)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import jensenshannon

from data.datasets import relation_answer_list
from facts.templates import render
from schema import BehavioralScores, Fact, History


def answer_token_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    answer: str,
    *,
    device: str,
) -> float:
    """Sum of answer-token log probabilities under teacher forcing."""
    import torch
    import torch.nn.functional as F

    prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    ans_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not ans_ids:
        ans_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not ans_ids:
        return float("-inf")

    input_ids = torch.cat(
        [
            prompt_ids["input_ids"],
            torch.tensor([ans_ids], dtype=prompt_ids["input_ids"].dtype),
        ],
        dim=1,
    ).to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    # Predict token t from position t-1.
    prompt_len = int(prompt_ids["input_ids"].shape[1])
    total = 0.0
    for i, tid in enumerate(ans_ids):
        pos = prompt_len + i - 1
        if pos < 0:
            continue
        log_probs = F.log_softmax(logits[0, pos, :], dim=-1)
        total += float(log_probs[tid].item())
    return total


def _answer_logprobs(
    model: Any, tokenizer: Any, prompt: str, answers: list[str], device: str
) -> np.ndarray:
    return np.asarray(
        [
            answer_token_logprob(model, tokenizer, prompt, ans, device=device)
            for ans in answers
        ],
        dtype=np.float64,
    )


def score_fact(
    model: Any,
    tokenizer: Any,
    fact: Fact,
    templates: list[str],
    *,
    history: History,
    device: str = "cpu",
) -> BehavioralScores:
    answers = relation_answer_list(fact.relation)
    all_probs = []
    ems = []
    for template in templates:
        prompt = render(template, fact)
        if prompt.endswith(fact.answer):
            prompt = prompt[: -len(fact.answer)].rstrip()
        logp = _answer_logprobs(model, tokenizer, prompt, answers, device)
        logp = logp - np.max(logp)
        probs = np.exp(logp)
        probs = probs / probs.sum()
        all_probs.append(probs)
        pred = answers[int(np.argmax(probs))]
        ems.append(1.0 if pred == fact.answer else 0.0)

    mean_probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    target_p = float(mean_probs[fact.answer_id])
    other = float(mean_probs.sum() - target_p)
    target_log_odds = float(np.log(target_p + 1e-12) - np.log(other + 1e-12))
    rank = int(1 + np.sum(mean_probs > mean_probs[fact.answer_id]))

    return BehavioralScores(
        fact_id=fact.fact_id,
        history=history,
        exact_match=float(np.mean(ems)),
        target_log_odds=target_log_odds,
        target_rank=rank,
        answer_probs=tuple(float(x) for x in mean_probs),
    )


def js_distance(p: tuple[float, ...] | np.ndarray, q: tuple[float, ...] | np.ndarray) -> float:
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    p_arr = p_arr / p_arr.sum()
    q_arr = q_arr / q_arr.sum()
    return float(jensenshannon(p_arr, q_arr, base=2) ** 2)


def target_aligned_probs(score: BehavioralScores, target_answer_id: int) -> np.ndarray:
    probs = np.asarray(score.answer_probs, dtype=np.float64)
    return np.roll(probs, -int(target_answer_id))


def _paired_bootstrap_ci(
    deltas: np.ndarray,
    *,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float,
) -> tuple[float, list[float]]:
    point = float(deltas.mean()) if deltas.size else 0.0
    if not deltas.size or n_boot <= 0:
        return point, [point, point]
    draws = np.array(
        [rng.choice(deltas, size=deltas.size, replace=True).mean() for _ in range(n_boot)]
    )
    lo, hi = alpha / 2, 1 - alpha / 2
    return point, [float(np.quantile(draws, lo)), float(np.quantile(draws, hi))]


def _mean_ci(
    values: np.ndarray, *, n_boot: int, rng: np.random.Generator, alpha: float
) -> tuple[float, list[float]]:
    point = float(values.mean()) if values.size else 0.0
    if not values.size or n_boot <= 0:
        return point, [point, point]
    draws = np.array(
        [rng.choice(values, size=values.size, replace=True).mean() for _ in range(n_boot)]
    )
    lo, hi = alpha / 2, 1 - alpha / 2
    return point, [float(np.quantile(draws, lo)), float(np.quantile(draws, hi))]


def aggregate_paired_fact_gaps(
    learned_by_fact: dict[str, BehavioralScores],
    control_by_fact: dict[str, BehavioralScores],
    *,
    target_answer_ids: dict[str, int] | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Within-fact learned-minus-control gaps (complementary design)."""
    common = sorted(set(learned_by_fact) & set(control_by_fact))
    if not common:
        return {}

    rng = np.random.default_rng(seed)
    ids = target_answer_ids or {}

    delta_lo = np.array(
        [
            learned_by_fact[fid].target_log_odds - control_by_fact[fid].target_log_odds
            for fid in common
        ],
        dtype=np.float64,
    )
    delta_em = np.array(
        [
            learned_by_fact[fid].exact_match - control_by_fact[fid].exact_match
            for fid in common
        ],
        dtype=np.float64,
    )
    js = np.array(
        [
            js_distance(
                target_aligned_probs(learned_by_fact[fid], ids.get(fid, 0)),
                target_aligned_probs(control_by_fact[fid], ids.get(fid, 0)),
            )
            for fid in common
        ],
        dtype=np.float64,
    )
    ranks_l = np.array([learned_by_fact[fid].target_rank for fid in common], dtype=np.float64)
    ranks_c = np.array([control_by_fact[fid].target_rank for fid in common], dtype=np.float64)

    dlo, dlo_ci = _paired_bootstrap_ci(delta_lo, n_boot=n_boot, rng=rng, alpha=alpha)
    dem, dem_ci = _paired_bootstrap_ci(delta_em, n_boot=n_boot, rng=rng, alpha=alpha)
    mean_js, mean_js_ci = _mean_ci(js, n_boot=n_boot, rng=rng, alpha=alpha)

    return {
        "delta_log_odds": dlo,
        "delta_log_odds_ci": dlo_ci,
        "delta_exact_match": dem,
        "delta_exact_match_ci": dem_ci,
        "mean_js": mean_js,
        "mean_js_ci": mean_js_ci,
        "mean_exact_match": float(
            np.mean(
                [
                    0.5
                    * (learned_by_fact[fid].exact_match + control_by_fact[fid].exact_match)
                    for fid in common
                ]
            )
        ),
        "learned_exact_match": float(np.mean([learned_by_fact[fid].exact_match for fid in common])),
        "control_exact_match": float(np.mean([control_by_fact[fid].exact_match for fid in common])),
        "learned_log_odds": float(
            np.mean([learned_by_fact[fid].target_log_odds for fid in common])
        ),
        "control_log_odds": float(
            np.mean([control_by_fact[fid].target_log_odds for fid in common])
        ),
        "mean_learned_rank": float(ranks_l.mean()),
        "mean_control_rank": float(ranks_c.mean()),
        "n_facts": int(len(common)),
        "n_boot": n_boot,
        "alpha": alpha,
    }


# Back-compat alias used by older call sites during the transition.
def aggregate_learned_control_gaps(
    scores: list[BehavioralScores],
    *,
    target_answer_ids: dict[str, int] | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    learned = {s.fact_id: s for s in scores if s.history == "learned"}
    control = {s.fact_id: s for s in scores if s.history == "control"}
    # If histories are disjoint within one complementary run, fall back to
    # unpaired group means (used only for within-run diagnostics).
    common = set(learned) & set(control)
    if common:
        return aggregate_paired_fact_gaps(
            learned,
            control,
            target_answer_ids=target_answer_ids,
            n_boot=n_boot,
            seed=seed,
            alpha=alpha,
        )
    if not learned or not control:
        return {}
    rng = np.random.default_rng(seed)
    lo_l = np.array([s.target_log_odds for s in learned.values()], dtype=np.float64)
    lo_c = np.array([s.target_log_odds for s in control.values()], dtype=np.float64)
    em_l = np.array([s.exact_match for s in learned.values()], dtype=np.float64)
    em_c = np.array([s.exact_match for s in control.values()], dtype=np.float64)
    point_lo = float(lo_l.mean() - lo_c.mean())
    point_em = float(em_l.mean() - em_c.mean())
    draws_lo = np.empty(n_boot, dtype=np.float64)
    draws_em = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        draws_lo[i] = (
            rng.choice(lo_l, size=lo_l.size, replace=True).mean()
            - rng.choice(lo_c, size=lo_c.size, replace=True).mean()
        )
        draws_em[i] = (
            rng.choice(em_l, size=em_l.size, replace=True).mean()
            - rng.choice(em_c, size=em_c.size, replace=True).mean()
        )
    lo, hi = alpha / 2, 1 - alpha / 2
    return {
        "delta_log_odds": point_lo,
        "delta_log_odds_ci": [float(np.quantile(draws_lo, lo)), float(np.quantile(draws_lo, hi))],
        "delta_exact_match": point_em,
        "delta_exact_match_ci": [
            float(np.quantile(draws_em, lo)),
            float(np.quantile(draws_em, hi)),
        ],
        "mean_js": 0.0,
        "mean_js_ci": [0.0, 0.0],
        "learned_exact_match": float(em_l.mean()),
        "control_exact_match": float(em_c.mean()),
        "learned_log_odds": float(lo_l.mean()),
        "control_log_odds": float(lo_c.mean()),
        "n_facts": int(min(lo_l.size, lo_c.size)),
        "n_boot": n_boot,
        "alpha": alpha,
        "warning": "unpaired_within_run",
    }
