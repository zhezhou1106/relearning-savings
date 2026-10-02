"""Shared Accelerate AdamW training loop."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from util.metrics import append_jsonl
from util.wandb_log import wandb_log


@dataclass
class TrainConfig:
    lr: float = 2e-5
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    effective_batch_size: int = 64
    micro_batch_size: int = 16
    max_seq_length: int = 256
    warmup_updates: int = 100
    total_updates: int | None = None
    lr_schedule: str = "cosine"


def build_optimizer(model: Any, cfg: TrainConfig) -> Any:
    import torch

    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )


def lr_at_step(step: int, cfg: TrainConfig) -> float:
    if cfg.total_updates is None or cfg.total_updates <= 0:
        return cfg.lr
    if step < cfg.warmup_updates:
        return cfg.lr * float(step + 1) / float(max(cfg.warmup_updates, 1))
    if cfg.lr_schedule != "cosine":
        return cfg.lr
    progress = (step - cfg.warmup_updates) / max(
        1, cfg.total_updates - cfg.warmup_updates
    )
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def train_steps(
    *,
    model: Any,
    tokenizer: Any,
    examples: Sequence[dict[str, Any]],
    cfg: TrainConfig,
    n_updates: int,
    metrics_path: str | None = None,
    collate_fn: Callable[..., dict[str, Any]] | None = None,
    start_step: int = 0,
    micro_batch_size: int | None = None,
    device: str | None = None,
    save_callback: Callable[[int, Any], None] | None = None,
    checkpoint_steps: Iterable[int] | None = None,
    wandb_log_every: int | None = 10,
    wandb_prefix: str = "train/",
    progress_desc: str = "train",
    show_progress: bool = True,
    optimizer: Any | None = None,
    shuffle: bool = True,
    shuffle_seed: int = 0,
    eval_callback: Callable[[int, Any], dict[str, Any] | None] | None = None,
    eval_every_updates: int | None = None,
) -> int:
    """Run n_updates of full-parameter fine-tuning; returns final step.

    `start_step` positions the run on the LR schedule, so a curve can be trained
    in segments without restarting warmup/decay. Pass `optimizer` to keep Adam
    moments across those segments.

    `eval_callback(step, model)` may return {"stop": True} to end training early
    (used for the Phase-A exact-match gate).
    """
    import sys

    import numpy as np
    import torch
    from data.collation import collate_texts
    from tqdm import tqdm

    if collate_fn is None:
        collate_fn = collate_texts

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    if optimizer is None:
        optimizer = build_optimizer(model, cfg)
    ckpt_set = set(checkpoint_steps or [])

    step = start_step
    idx = 0
    n = len(examples)
    if n == 0:
        raise ValueError("No training examples")

    # Epoch-wise shuffling: without it the fixed cycle puts every template of one
    # fact (and both of its history blocks) into the same batch.
    rng = np.random.default_rng(shuffle_seed)
    order = rng.permutation(n) if shuffle else np.arange(n)

    def next_index() -> int:
        nonlocal idx, order
        if idx >= n:
            order = rng.permutation(n) if shuffle else np.arange(n)
            idx = 0
        value = int(order[idx])
        idx += 1
        return value

    mb = max(1, int(micro_batch_size if micro_batch_size is not None else cfg.micro_batch_size))
    mb = min(mb, max(1, cfg.effective_batch_size))
    accum = max(1, cfg.effective_batch_size // mb)
    log_every = None if not wandb_log_every or wandb_log_every <= 0 else int(wandb_log_every)
    prefix = wandb_prefix or ""

    # Non-TTY logs: emit newline updates; mininterval keeps the log readable.
    pbar = tqdm(
        total=n_updates,
        desc=progress_desc,
        initial=0,
        disable=not show_progress,
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=10.0,
        smoothing=0.05,
    )
    try:
        while step < start_step + n_updates:
            lr = lr_at_step(step, cfg)
            for g in optimizer.param_groups:
                g["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            last_loss = None
            for _ in range(accum):
                batch_ex = [examples[next_index()] for _ in range(mb)]
                # Default collate_texts accepts dicts or strings; answer-token
                # collators always take example dicts.
                batch = collate_fn(batch_ex, tokenizer, max_length=cfg.max_seq_length)
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                loss = out.loss / accum
                loss.backward()
                last_loss = loss

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            step += 1
            loss_val = (
                float(last_loss.detach().item() * accum) if last_loss is not None else None
            )
            if metrics_path is not None and loss_val is not None:
                append_jsonl(
                    metrics_path,
                    {"step": step, "loss": loss_val},
                )
            if log_every is not None and loss_val is not None:
                wandb_log(
                    {f"{prefix}loss": loss_val, f"{prefix}lr": lr},
                    step=step,
                    every=log_every,
                )
            if save_callback is not None and step in ckpt_set:
                save_callback(step, model)
            if loss_val is not None:
                pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr:.2e}", refresh=False)
            pbar.update(1)

            if (
                eval_callback is not None
                and eval_every_updates
                and step % int(eval_every_updates) == 0
            ):
                verdict = eval_callback(step, model)
                model.train()
                if verdict and verdict.get("stop"):
                    break
    finally:
        pbar.close()

    return step


def load_model_and_tokenizer(model_id: str, torch_dtype: str = "bfloat16"):
    """Load a model/tokenizer pair at an explicit dtype.

    Every stage goes through here so Phase A, Phase B, eval, probing and
    relearning all run the model at the same precision.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    return model, tokenizer
