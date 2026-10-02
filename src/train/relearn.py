"""Per-fact relearning curves with a fresh AdamW optimizer (workshop config)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from data.collation import collate_answer_only
from data.datasets import build_relearn_examples, relation_answer_list
from eval.behavioral import answer_token_logprob
from facts.templates import render
from schema import Fact
from train.trainer import TrainConfig, train_steps
from util.metrics import write_json
from util.paths import ensure_dir


def _target_log_odds(
    model: Any, tokenizer: Any, fact: Fact, eval_templates: list[str], device: str
) -> float:
    was_training = model.training
    model.eval()
    answers = relation_answer_list(fact.relation)
    log_odds_vals: list[float] = []
    for template in eval_templates:
        prompt = render(template, fact).rstrip()
        if prompt.endswith(fact.answer):
            prompt = prompt[: -len(fact.answer)].rstrip()
        scores = np.array(
            [
                answer_token_logprob(model, tokenizer, prompt, ans, device=device)
                for ans in answers
            ],
            dtype=np.float64,
        )
        target = scores[fact.answer_id]
        others = np.logaddexp.reduce(np.delete(scores, fact.answer_id))
        log_odds_vals.append(float(target - others))
    if was_training:
        model.train()
    return float(np.mean(log_odds_vals))


def load_relearn_checkpoint(
    ckpt_path: str | Path, device: str | None = None, torch_dtype: str = "bfloat16"
):
    """Load model/tokenizer once per checkpoint; keep a restore snapshot."""
    import torch

    from train.trainer import load_model_and_tokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(str(ckpt_path), torch_dtype=torch_dtype)
    model.to(device)
    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return model, tokenizer, base_state, device


def relearn_fact_curve(
    *,
    fact: Fact,
    ckpt_path: str | Path | None = None,
    train_templates: list[str],
    eval_templates: list[str],
    eval_ks: list[int],
    train_cfg: TrainConfig,
    out_dir: str | Path,
    dry_run: bool = False,
    model: Any | None = None,
    tokenizer: Any | None = None,
    base_state: dict[str, Any] | None = None,
    device: str | None = None,
    n_train_templates: int | None = None,
    n_eval_templates: int | None = None,
    target_prob_threshold: float = 0.90,
    answer_token_only_loss: bool = True,
) -> list[dict[str, Any]]:
    out = ensure_dir(out_dir)
    if n_train_templates is not None:
        train_templates = train_templates[:n_train_templates]
    if n_eval_templates is not None:
        eval_templates = eval_templates[:n_eval_templates]
    examples = build_relearn_examples(fact, train_templates)
    if dry_run:
        rows = [
            {
                "fact_id": fact.fact_id,
                "k": k,
                "target_log_odds": 0.0,
                "target_prob": 0.5,
                "gain": 0.0,
                "exact_match": 0.0,
                "reached_criterion": False,
            }
            for k in eval_ks
        ]
        write_json(out / "curve.json", rows)
        return rows

    owns_model = model is None
    if owns_model:
        if ckpt_path is None:
            raise ValueError("ckpt_path is required when model is not provided")
        model, tokenizer, base_state, device = load_relearn_checkpoint(ckpt_path, device)
    elif tokenizer is None or base_state is None:
        raise ValueError("tokenizer and base_state are required when model is provided")
    else:
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.load_state_dict(base_state)

    rows: list[dict[str, Any]] = []
    base_odds = None
    # Workshop: constant LR, WD=0, batch of both relearn templates per update.
    batch_size = max(1, int(getattr(train_cfg, "effective_batch_size", 2) or 2))
    cfg = TrainConfig(
        lr=train_cfg.lr,
        betas=train_cfg.betas,
        weight_decay=train_cfg.weight_decay,
        grad_clip=train_cfg.grad_clip,
        effective_batch_size=batch_size,
        micro_batch_size=batch_size,
        max_seq_length=train_cfg.max_seq_length,
        warmup_updates=0,
        total_updates=max(eval_ks),
        lr_schedule="constant",
    )

    from train.trainer import build_optimizer

    model.to(device)
    model.train()
    optimizer = build_optimizer(model, cfg)
    collate_fn = collate_answer_only if answer_token_only_loss else None

    prev_k = 0
    for k in eval_ks:
        delta = k - prev_k
        if delta > 0:
            train_steps(
                model=model,
                tokenizer=tokenizer,
                examples=examples,
                cfg=cfg,
                n_updates=delta,
                start_step=prev_k,
                optimizer=optimizer,
                device=device,
                micro_batch_size=batch_size,
                shuffle=True,
                shuffle_seed=abs(hash(fact.fact_id)) % (2**31),
                wandb_log_every=None,
                show_progress=False,
                collate_fn=collate_fn,
            )
        odds = _target_log_odds(model, tokenizer, fact, eval_templates, device)
        if base_odds is None:
            base_odds = odds
        target_prob = float(1.0 / (1.0 + np.exp(-odds)))
        rows.append(
            {
                "fact_id": fact.fact_id,
                "k": k,
                "target_log_odds": odds,
                "baseline_log_odds": base_odds,
                "target_prob": target_prob,
                "gain": odds - base_odds,
                "reached_criterion": bool(target_prob >= target_prob_threshold),
            }
        )
        prev_k = k

    write_json(out / "curve.json", rows)
    del optimizer
    if owns_model:
        del model
    return rows
