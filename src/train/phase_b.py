"""Phase B: rehearsal-free continual fine-tuning on a new fact stream."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from data.collation import collate_answer_only
from data.datasets import build_phase_b_stream
from eval.stream_health import log_stream_health, measure_stream_health
from schema import Fact
from train.trainer import TrainConfig, load_model_and_tokenizer, train_steps
from util.metrics import write_json
from util.paths import ensure_dir
from util.wandb_log import wandb_log


def run_phase_b(
    *,
    phase_a_ckpt: str | Path,
    phase_a_templates: dict[str, list[str]],
    out_dir: str | Path,
    train_cfg: TrainConfig,
    total_updates: int = 8000,
    n_stream_facts: int = 3000,
    checkpoint_steps: list[int] | None = None,
    seed: int = 123,
    dry_run: bool = False,
    exclude_entities: Iterable[str] | None = None,
    model_dtype: str = "bfloat16",
    general_texts: list[str] | None = None,
    format_prompts: list[str] | None = None,
    panel_facts: list[Fact] | None = None,
    overwrite_panel: bool = False,
    overwrite_scope: str | None = None,
    overwrite_relation: str | None = None,
    answer_token_only_loss: bool = False,
    n_templates_per_fact: int = 2,
) -> dict[str, Any]:
    out = ensure_dir(out_dir)
    steps = checkpoint_steps or [0, 250, 500, 1000, 2000, 4000, 8000]
    excluded = set(exclude_entities or ())
    if not excluded:
        raise ValueError(
            "exclude_entities must list the Phase-A panel and answer-exposure "
            "entities so the stream stays rehearsal-free"
        )
    examples = build_phase_b_stream(
        n_facts=n_stream_facts,
        seed=seed,
        phase_a_templates=phase_a_templates,
        exclude_entities=excluded,
        panel_facts=panel_facts,
        overwrite_panel=overwrite_panel,
        overwrite_scope=overwrite_scope,
        overwrite_relation=overwrite_relation,
        n_templates_per_fact=n_templates_per_fact,
    )
    scope = overwrite_scope
    if scope is None:
        scope = "full" if overwrite_panel else "none"
    n_overwrite = sum(1 for e in examples if e.get("overwrite"))
    meta = {
        "n_stream_examples": len(examples),
        "n_overwrite_examples": n_overwrite,
        "overwrite_panel": overwrite_panel,
        "overwrite_scope": scope,
        "overwrite_relation": overwrite_relation,
        "answer_token_only_loss": answer_token_only_loss,
        "total_updates": total_updates,
        "checkpoint_steps": steps,
        "phase_a_ckpt": str(phase_a_ckpt),
        "n_excluded_entities": len(excluded),
        "dry_run": dry_run,
    }
    write_json(out / "phase_b_meta.json", meta)
    if dry_run:
        write_json(out / "phase_b_examples_sample.json", examples[:20])
        for s in steps:
            ensure_dir(out / f"ckpt_{s}")
        return meta

    model, tokenizer = load_model_and_tokenizer(str(phase_a_ckpt), model_dtype)
    # The A/B boundary always starts a fresh optimizer and schedule: Phase A's
    # Adam state is not carried over, and total_updates changes here.
    cfg = TrainConfig(
        lr=train_cfg.lr,
        betas=train_cfg.betas,
        weight_decay=train_cfg.weight_decay,
        grad_clip=train_cfg.grad_clip,
        effective_batch_size=train_cfg.effective_batch_size,
        micro_batch_size=train_cfg.micro_batch_size,
        max_seq_length=train_cfg.max_seq_length,
        warmup_updates=train_cfg.warmup_updates,
        total_updates=total_updates,
        lr_schedule=train_cfg.lr_schedule,
    )

    # Save step-0 checkpoint before stream updates.
    ensure_dir(out / "ckpt_0")
    model.save_pretrained(out / "ckpt_0")
    tokenizer.save_pretrained(out / "ckpt_0")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    health_path = str(out / "metrics" / "stream_health.jsonl")
    # Held-out slice of the stream: never trained on, so its loss separates
    # learning the stream from memorising the exact examples.
    heldout = [e["text"] for e in examples[::37]][:64]
    collate_fn = collate_answer_only if answer_token_only_loss else None

    def _health(step: int, m: Any) -> None:
        if not general_texts or not format_prompts:
            return
        stats = measure_stream_health(
            m,
            tokenizer,
            stream_texts=heldout,
            general_texts=general_texts,
            format_prompts=format_prompts,
            device=device,
        )
        record = log_stream_health(health_path, stream_steps=step, **stats)
        wandb_log({f"stream_health/{k}": v for k, v in record.items()}, step=step)
        print(
            f"[stream health] step {step}: general_ppl={record['general_ppl']:.2f} "
            f"stream_ppl={record['stream_ppl']:.2f} "
            f"format={record['format_adherence_rate']:.2f}",
            flush=True,
        )

    def _save(step: int, m: Any) -> None:
        d = ensure_dir(out / f"ckpt_{step}")
        m.save_pretrained(d)
        tokenizer.save_pretrained(d)
        _health(step, m)
        m.train()

    model.to(device)
    _health(0, model)

    train_steps(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        cfg=cfg,
        n_updates=total_updates,
        metrics_path=str(out / "metrics" / "phase_b_stream.jsonl"),
        save_callback=_save,
        checkpoint_steps=[s for s in steps if s > 0],
        progress_desc="phase_b",
        shuffle_seed=seed,
        collate_fn=collate_fn,
    )
    write_json(out / "phase_b_meta.json", meta)
    return meta
