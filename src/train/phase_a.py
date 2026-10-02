"""Phase A: establish learned / matched-control histories with joint stopping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data.collation import collate_answer_only
from data.datasets import build_phase_a_examples
from schema import Fact
from train.trainer import TrainConfig, load_model_and_tokenizer, train_steps
from util.metrics import append_jsonl, write_json
from util.paths import ensure_dir
from util.wandb_log import wandb_log

GATE_HISTORIES = ("learned", "control", "wrong")


def trained_target(fact: Fact, assignment: dict[str, Any]) -> Fact:
    """Binding Phase A actually trains for this fact under its history."""
    return Fact(
        fact_id=fact.fact_id,
        relation=str(assignment["train_relation"]),
        entity=fact.entity,
        answer=str(assignment["train_answer"]),
        answer_id=int(assignment["train_answer_id"]),
    )


def evaluate_phase_a_gate(
    model: Any,
    tokenizer: Any,
    facts: list[Fact],
    assignments: dict[str, dict[str, Any]],
    behavioral_templates: dict[str, list[str]],
    *,
    n_per_history: int = 96,
    device: str | None = None,
    fact_ids: list[str] | None = None,
) -> dict[str, float]:
    """Phase-A manipulation checks on development facts."""
    from eval.behavioral import score_fact

    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    facts_by_id = {f.fact_id: f for f in facts}
    allowed = set(fact_ids) if fact_ids is not None else None

    by_history: dict[str, list[str]] = {h: [] for h in GATE_HISTORIES}
    for fid in sorted(facts_by_id):
        if allowed is not None and fid not in allowed:
            continue
        history = assignments[fid]["history"]
        if history in by_history and len(by_history[history]) < n_per_history:
            by_history[history].append(fid)

    was_training = model.training
    model.eval()
    scores: dict[str, float] = {}
    learned_target_lo: list[float] = []
    control_target_lo: list[float] = []
    wrong_target_lo: list[float] = []
    try:
        for history, ids in by_history.items():
            if not ids:
                continue
            ems = []
            for fid in ids:
                proxy = trained_target(facts_by_id[fid], assignments[fid])
                result = score_fact(
                    model,
                    tokenizer,
                    proxy,
                    behavioral_templates[proxy.relation],
                    history=history,  # type: ignore[arg-type]
                    device=device,
                )
                ems.append(result.exact_match)
                # Target-binding residual: score the true target fact, not the
                # registry/wrong proxy, for residual and separation checks.
                target_result = score_fact(
                    model,
                    tokenizer,
                    facts_by_id[fid],
                    behavioral_templates[facts_by_id[fid].relation],
                    history=history,  # type: ignore[arg-type]
                    device=device,
                )
                if history == "learned":
                    learned_target_lo.append(target_result.target_log_odds)
                elif history == "control":
                    control_target_lo.append(target_result.target_log_odds)
                elif history == "wrong":
                    wrong_target_lo.append(target_result.target_log_odds)
            scores[history] = float(sum(ems) / len(ems))
    finally:
        if was_training:
            model.train()

    scores["n_per_history"] = float(min((len(v) for v in by_history.values()), default=0))
    scores["learned_target_log_odds"] = (
        float(sum(learned_target_lo) / len(learned_target_lo)) if learned_target_lo else 0.0
    )
    scores["control_target_log_odds"] = (
        float(sum(control_target_lo) / len(control_target_lo)) if control_target_lo else 0.0
    )
    scores["wrong_target_log_odds"] = (
        float(sum(wrong_target_lo) / len(wrong_target_lo)) if wrong_target_lo else 0.0
    )
    scores["log_odds_separation"] = (
        scores["learned_target_log_odds"] - scores["control_target_log_odds"]
    )
    return scores


def gate_passed(
    scores: dict[str, float],
    *,
    target_exact_match: float,
    min_log_odds_separation: float,
    control_residual_log_odds_max: float,
    wrong_residual_log_odds_max: float | None = None,
    require_wrong: bool | None = None,
) -> bool:
    """Joint Phase-A manipulation checks.

    When wrong-arm scores are present (or ``require_wrong`` is True), also
    require W exact-match on a' and strongly negative residual LO on true a.
    """
    has_wrong = "wrong" in scores or bool(require_wrong)
    wrong_ok = True
    if has_wrong:
        wr_max = (
            wrong_residual_log_odds_max
            if wrong_residual_log_odds_max is not None
            else -1.0
        )
        wrong_ok = (
            scores.get("wrong", 0.0) >= target_exact_match
            and scores.get("wrong_target_log_odds", 1e9) <= wr_max
        )
    return (
        scores.get("learned", 0.0) >= target_exact_match
        and scores.get("control", 0.0) >= target_exact_match
        and scores.get("log_odds_separation", -1e9) >= min_log_odds_separation
        and scores.get("control_target_log_odds", 1e9) <= control_residual_log_odds_max
        and wrong_ok
    )


def run_phase_a(
    *,
    facts: list[Fact],
    assignments: dict[str, dict[str, Any]],
    phase_a_templates: dict[str, list[str]],
    model_id: str,
    out_dir: str | Path,
    train_cfg: TrainConfig,
    torch_dtype: str = "bfloat16",
    max_updates: int = 5000,
    dry_run: bool = False,
    behavioral_templates: dict[str, list[str]] | None = None,
    target_exact_match: float = 0.90,
    min_log_odds_separation: float = 1.0,
    control_residual_log_odds_max: float = -0.5,
    eval_every_updates: int = 250,
    eval_facts_per_history: int = 96,
    stop_on_target: bool = True,
    require_target: bool = True,
    seed: int = 0,
    gate_fact_ids: list[str] | None = None,
    max_train_updates: int | None = None,
    answer_token_only_loss: bool = True,
) -> dict[str, Any]:
    """Train one complementary trajectory (used inside joint pair training)."""
    out = ensure_dir(out_dir)
    examples = build_phase_a_examples(facts, assignments, phase_a_templates)
    n_updates = max_train_updates if max_train_updates is not None else max_updates
    collate_fn = collate_answer_only if answer_token_only_loss else None
    meta = {
        "n_facts": len(facts),
        "n_examples": len(examples),
        "model_id": model_id,
        "target_exact_match": target_exact_match,
        "dry_run": dry_run,
        "max_train_updates": n_updates,
        "answer_token_only_loss": answer_token_only_loss,
    }
    write_json(out / "phase_a_meta.json", meta)

    if dry_run:
        write_json(out / "phase_a_examples_sample.json", examples[:20])
        ensure_dir(out / "checkpoint")
        return meta

    model, tokenizer = load_model_and_tokenizer(model_id, torch_dtype=torch_dtype)
    cfg = TrainConfig(
        lr=train_cfg.lr,
        betas=train_cfg.betas,
        weight_decay=train_cfg.weight_decay,
        grad_clip=train_cfg.grad_clip,
        effective_batch_size=train_cfg.effective_batch_size,
        micro_batch_size=train_cfg.micro_batch_size,
        max_seq_length=train_cfg.max_seq_length,
        warmup_updates=train_cfg.warmup_updates,
        total_updates=max_updates,
        lr_schedule=train_cfg.lr_schedule,
    )
    ckpt_dir = ensure_dir(out / "checkpoint")

    def _save(step: int, m: Any) -> None:
        m.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

    gate_history: list[dict[str, Any]] = []
    eval_callback = None
    if behavioral_templates is not None and eval_every_updates > 0:

        def eval_callback(step: int, m: Any) -> dict[str, Any] | None:
            scores = evaluate_phase_a_gate(
                m,
                tokenizer,
                facts,
                assignments,
                behavioral_templates,
                n_per_history=eval_facts_per_history,
                fact_ids=gate_fact_ids,
            )
            record = {"step": step, **scores}
            gate_history.append(record)
            append_jsonl(str(out / "metrics" / "phase_a_gate.jsonl"), record)
            wandb_log({f"phase_a_gate/{k}": v for k, v in scores.items()}, step=step)
            reached = gate_passed(
                scores,
                target_exact_match=target_exact_match,
                min_log_odds_separation=min_log_odds_separation,
                control_residual_log_odds_max=control_residual_log_odds_max,
            )
            print(
                f"[phase_a gate] step {step}: "
                + ", ".join(f"{h}={scores.get(h, float('nan')):.3f}" for h in GATE_HISTORIES)
                + f" sep={scores.get('log_odds_separation', float('nan')):.3f}"
                + f" ctrl_lo={scores.get('control_target_log_odds', float('nan')):.3f}"
                + f" (target {target_exact_match:.2f}{', reached' if reached else ''})",
                flush=True,
            )
            return {"stop": True} if (reached and stop_on_target) else None

    final_step = train_steps(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        cfg=cfg,
        n_updates=n_updates,
        metrics_path=str(out / "metrics" / "phase_a.jsonl"),
        save_callback=_save,
        checkpoint_steps=[n_updates],
        progress_desc="phase_a",
        shuffle_seed=seed,
        eval_callback=eval_callback,
        eval_every_updates=eval_every_updates,
        collate_fn=collate_fn,
    )
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    final_scores: dict[str, float] = {}
    if behavioral_templates is not None:
        final_scores = evaluate_phase_a_gate(
            model,
            tokenizer,
            facts,
            assignments,
            behavioral_templates,
            n_per_history=eval_facts_per_history,
            fact_ids=gate_fact_ids,
        )
    meta["checkpoint"] = str(ckpt_dir)
    meta["final_step"] = final_step
    meta["gate_history"] = gate_history
    meta["final_scores"] = final_scores
    meta["target_reached"] = bool(
        final_scores
        and gate_passed(
            final_scores,
            target_exact_match=target_exact_match,
            min_log_odds_separation=min_log_odds_separation,
            control_residual_log_odds_max=control_residual_log_odds_max,
        )
    )
    write_json(out / "phase_a_meta.json", meta)

    if require_target and final_scores and not meta["target_reached"]:
        raise RuntimeError(
            f"Phase A did not reach the workshop gate criteria: {final_scores}"
        )
    return meta


def run_phase_a_pair(
    *,
    facts: list[Fact],
    assignments_c0: dict[str, dict[str, Any]],
    assignments_c1: dict[str, dict[str, Any]],
    phase_a_templates: dict[str, list[str]],
    behavioral_templates: dict[str, list[str]],
    model_id: str,
    out_dir_c0: str | Path,
    out_dir_c1: str | Path,
    train_cfg: TrainConfig,
    torch_dtype: str = "bfloat16",
    max_updates: int = 8000,
    dry_run: bool = False,
    target_exact_match: float = 0.90,
    min_log_odds_separation: float = 1.0,
    control_residual_log_odds_max: float = -0.5,
    eval_every_updates: int = 250,
    eval_facts_per_history: int = 96,
    require_target: bool = True,
    seed: int = 0,
    gate_fact_ids: list[str] | None = None,
    answer_token_only_loss: bool = True,
) -> dict[str, Any]:
    """Joint complementary Phase A: stop at first shared update both pass."""
    out0 = ensure_dir(out_dir_c0)
    out1 = ensure_dir(out_dir_c1)
    examples0 = build_phase_a_examples(facts, assignments_c0, phase_a_templates)
    examples1 = build_phase_a_examples(facts, assignments_c1, phase_a_templates)
    collate_fn = collate_answer_only if answer_token_only_loss else None

    joint_meta: dict[str, Any] = {
        "n_facts": len(facts),
        "n_examples_c0": len(examples0),
        "n_examples_c1": len(examples1),
        "model_id": model_id,
        "dry_run": dry_run,
        "eval_every_updates": eval_every_updates,
        "answer_token_only_loss": answer_token_only_loss,
    }
    if dry_run:
        write_json(out0 / "phase_a_meta.json", {**joint_meta, "complementary": 0})
        write_json(out1 / "phase_a_meta.json", {**joint_meta, "complementary": 1})
        ensure_dir(out0 / "checkpoint")
        ensure_dir(out1 / "checkpoint")
        write_json(out0.parent / "phase_a_joint_meta.json", joint_meta)
        return joint_meta

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model0, tokenizer = load_model_and_tokenizer(model_id, torch_dtype=torch_dtype)
    model1, _ = load_model_and_tokenizer(model_id, torch_dtype=torch_dtype)
    cfg = TrainConfig(
        lr=train_cfg.lr,
        betas=train_cfg.betas,
        weight_decay=train_cfg.weight_decay,
        grad_clip=train_cfg.grad_clip,
        effective_batch_size=train_cfg.effective_batch_size,
        micro_batch_size=train_cfg.micro_batch_size,
        max_seq_length=train_cfg.max_seq_length,
        warmup_updates=train_cfg.warmup_updates,
        total_updates=max_updates,
        lr_schedule=train_cfg.lr_schedule,
    )

    from train.trainer import build_optimizer

    opt0 = build_optimizer(model0, cfg)
    opt1 = build_optimizer(model1, cfg)
    model0.to(device)
    model1.to(device)

    gate_history: list[dict[str, Any]] = []
    step = 0
    stop_step: int | None = None

    while step < max_updates:
        block = min(eval_every_updates, max_updates - step)
        step = train_steps(
            model=model0,
            tokenizer=tokenizer,
            examples=examples0,
            cfg=cfg,
            n_updates=block,
            start_step=step,
            optimizer=opt0,
            device=device,
            shuffle_seed=seed + step,
            progress_desc=f"phase_a_c0@{step}",
            show_progress=True,
            wandb_log_every=None,
            collate_fn=collate_fn,
        )
        # Mirror the same number of updates on c1 from the same schedule position.
        train_steps(
            model=model1,
            tokenizer=tokenizer,
            examples=examples1,
            cfg=cfg,
            n_updates=block,
            start_step=step - block,
            optimizer=opt1,
            device=device,
            shuffle_seed=seed + 10_000 + (step - block),
            progress_desc=f"phase_a_c1@{step - block}",
            show_progress=True,
            wandb_log_every=None,
            collate_fn=collate_fn,
        )

        scores0 = evaluate_phase_a_gate(
            model0,
            tokenizer,
            facts,
            assignments_c0,
            behavioral_templates,
            n_per_history=eval_facts_per_history,
            fact_ids=gate_fact_ids,
            device=device,
        )
        scores1 = evaluate_phase_a_gate(
            model1,
            tokenizer,
            facts,
            assignments_c1,
            behavioral_templates,
            n_per_history=eval_facts_per_history,
            fact_ids=gate_fact_ids,
            device=device,
        )
        record = {"step": step, "c0": scores0, "c1": scores1}
        gate_history.append(record)
        append_jsonl(str(out0.parent / "phase_a_joint_gate.jsonl"), record)
        wandb_log(
            {
                **{f"phase_a_c0/{k}": v for k, v in scores0.items()},
                **{f"phase_a_c1/{k}": v for k, v in scores1.items()},
            },
            step=step,
        )
        ok0 = gate_passed(
            scores0,
            target_exact_match=target_exact_match,
            min_log_odds_separation=min_log_odds_separation,
            control_residual_log_odds_max=control_residual_log_odds_max,
        )
        ok1 = gate_passed(
            scores1,
            target_exact_match=target_exact_match,
            min_log_odds_separation=min_log_odds_separation,
            control_residual_log_odds_max=control_residual_log_odds_max,
        )
        print(
            f"[phase_a joint] step {step}: c0_ok={ok0} c1_ok={ok1} "
            f"c0_sep={scores0.get('log_odds_separation', float('nan')):.3f} "
            f"c1_sep={scores1.get('log_odds_separation', float('nan')):.3f}",
            flush=True,
        )
        if ok0 and ok1:
            stop_step = step
            break

    ckpt0 = ensure_dir(out0 / "checkpoint")
    ckpt1 = ensure_dir(out1 / "checkpoint")
    model0.save_pretrained(ckpt0)
    tokenizer.save_pretrained(ckpt0)
    model1.save_pretrained(ckpt1)
    tokenizer.save_pretrained(ckpt1)

    joint_meta.update(
        {
            "final_step": stop_step if stop_step is not None else step,
            "target_reached": stop_step is not None,
            "gate_history": gate_history,
            "checkpoint_c0": str(ckpt0),
            "checkpoint_c1": str(ckpt1),
        }
    )
    write_json(out0 / "phase_a_meta.json", {**joint_meta, "complementary": 0})
    write_json(out1 / "phase_a_meta.json", {**joint_meta, "complementary": 1})
    write_json(out0.parent / "phase_a_joint_meta.json", joint_meta)

    if require_target and stop_step is None:
        raise RuntimeError(
            f"Joint Phase A did not reach the workshop gate within {max_updates} updates"
        )
    return joint_meta


def run_phase_a_group(
    *,
    facts: list[Fact],
    assignments_by_run: dict[int, dict[str, dict[str, Any]]],
    phase_a_templates: dict[str, list[str]],
    behavioral_templates: dict[str, list[str]],
    model_id: str,
    out_dirs: dict[int, str | Path],
    train_cfg: TrainConfig,
    torch_dtype: str = "bfloat16",
    max_updates: int = 8000,
    dry_run: bool = False,
    target_exact_match: float = 0.90,
    min_log_odds_separation: float = 1.0,
    control_residual_log_odds_max: float = -0.5,
    wrong_residual_log_odds_max: float = -1.0,
    eval_every_updates: int = 250,
    eval_facts_per_history: int = 96,
    require_target: bool = True,
    seed: int = 0,
    gate_fact_ids: list[str] | None = None,
    answer_token_only_loss: bool = True,
) -> dict[str, Any]:
    """Three-arm Phase A with a two-pass joint stop (one model resident at a time).

    Pass 1: train each run independently, saving a checkpoint at every gate
    eval and recording the earliest passing step. Pass 2: take
    K* = max_r earliest_pass_r, promote the K* checkpoint for every run to the
    final ``checkpoint/`` directory, and discard the rest. Semantically matches
    the two-arm "first shared update at which all runs pass" rule without
    holding three full models in GPU memory.
    """
    import shutil

    import torch

    from train.trainer import build_optimizer

    runs = sorted(assignments_by_run)
    outs = {r: ensure_dir(out_dirs[r]) for r in runs}
    collate_fn = collate_answer_only if answer_token_only_loss else None
    examples = {
        r: build_phase_a_examples(facts, assignments_by_run[r], phase_a_templates)
        for r in runs
    }
    joint_meta: dict[str, Any] = {
        "n_facts": len(facts),
        "n_runs": len(runs),
        "n_examples": {str(r): len(examples[r]) for r in runs},
        "model_id": model_id,
        "dry_run": dry_run,
        "eval_every_updates": eval_every_updates,
        "answer_token_only_loss": answer_token_only_loss,
        "wrong_residual_log_odds_max": wrong_residual_log_odds_max,
    }
    seed_root = outs[runs[0]].parent
    if dry_run:
        for r in runs:
            write_json(outs[r] / "phase_a_meta.json", {**joint_meta, "complementary": r})
            ensure_dir(outs[r] / "checkpoint")
        write_json(seed_root / "phase_a_joint_meta.json", joint_meta)
        return joint_meta

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TrainConfig(
        lr=train_cfg.lr,
        betas=train_cfg.betas,
        weight_decay=train_cfg.weight_decay,
        grad_clip=train_cfg.grad_clip,
        effective_batch_size=train_cfg.effective_batch_size,
        micro_batch_size=train_cfg.micro_batch_size,
        max_seq_length=train_cfg.max_seq_length,
        warmup_updates=train_cfg.warmup_updates,
        total_updates=max_updates,
        lr_schedule=train_cfg.lr_schedule,
    )

    earliest_pass: dict[int, int | None] = {r: None for r in runs}
    gate_history: dict[int, list[dict[str, Any]]] = {r: [] for r in runs}

    # ---- Pass 1: train each run independently ----
    for r in runs:
        print(f"[phase_a group] pass-1 training run c{r}", flush=True)
        model, tokenizer = load_model_and_tokenizer(model_id, torch_dtype=torch_dtype)
        model.to(device)
        opt = build_optimizer(model, cfg)
        step = 0
        while step < max_updates:
            block = min(eval_every_updates, max_updates - step)
            step = train_steps(
                model=model,
                tokenizer=tokenizer,
                examples=examples[r],
                cfg=cfg,
                n_updates=block,
                start_step=step,
                optimizer=opt,
                device=device,
                shuffle_seed=seed + 1000 * r + step,
                progress_desc=f"phase_a_c{r}@{step}",
                show_progress=True,
                wandb_log_every=None,
                collate_fn=collate_fn,
            )
            scores = evaluate_phase_a_gate(
                model,
                tokenizer,
                facts,
                assignments_by_run[r],
                behavioral_templates,
                n_per_history=eval_facts_per_history,
                fact_ids=gate_fact_ids,
                device=device,
            )
            record = {"step": step, **scores}
            gate_history[r].append(record)
            append_jsonl(str(outs[r] / "metrics" / "phase_a_gate.jsonl"), record)
            wandb_log({f"phase_a_c{r}/{k}": v for k, v in scores.items()}, step=step)
            ok = gate_passed(
                scores,
                target_exact_match=target_exact_match,
                min_log_odds_separation=min_log_odds_separation,
                control_residual_log_odds_max=control_residual_log_odds_max,
                wrong_residual_log_odds_max=wrong_residual_log_odds_max,
                require_wrong=True,
            )
            # Save intermediate checkpoint at every gate eval.
            ckpt_step = ensure_dir(outs[r] / f"ckpt_{step}")
            model.save_pretrained(ckpt_step)
            tokenizer.save_pretrained(ckpt_step)
            print(
                f"[phase_a group] c{r} step {step}: ok={ok} "
                f"L={scores.get('learned', float('nan')):.3f} "
                f"C={scores.get('control', float('nan')):.3f} "
                f"W={scores.get('wrong', float('nan')):.3f} "
                f"sep={scores.get('log_odds_separation', float('nan')):.3f} "
                f"W_lo={scores.get('wrong_target_log_odds', float('nan')):.3f}",
                flush=True,
            )
            if ok and earliest_pass[r] is None:
                earliest_pass[r] = step
                break
        del model, opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if any(v is None for v in earliest_pass.values()):
        joint_meta.update(
            {
                "final_step": None,
                "target_reached": False,
                "earliest_pass": {str(k): v for k, v in earliest_pass.items()},
                "gate_history": {str(k): v for k, v in gate_history.items()},
            }
        )
        write_json(seed_root / "phase_a_joint_meta.json", joint_meta)
        if require_target:
            raise RuntimeError(
                f"Three-arm Phase A did not reach the gate within {max_updates} "
                f"updates: earliest_pass={earliest_pass}"
            )
        return joint_meta

    # ---- Pass 2: promote K* = max earliest_pass ----
    k_star = max(int(v) for v in earliest_pass.values() if v is not None)
    print(
        f"[phase_a group] pass-2: K*={k_star} from earliest_pass={earliest_pass}",
        flush=True,
    )
    for r in runs:
        src = outs[r] / f"ckpt_{k_star}"
        if not src.exists():
            # Run may have stopped earlier; continue training from its last
            # checkpoint up to K* (rare: only if another run needed more steps).
            last = earliest_pass[r]
            assert last is not None
            if last < k_star:
                print(
                    f"[phase_a group] c{r} continuing {last} → {k_star}",
                    flush=True,
                )
                model, tokenizer = load_model_and_tokenizer(
                    str(outs[r] / f"ckpt_{last}"), torch_dtype=torch_dtype
                )
                model.to(device)
                opt = build_optimizer(model, cfg)
                train_steps(
                    model=model,
                    tokenizer=tokenizer,
                    examples=examples[r],
                    cfg=cfg,
                    n_updates=k_star - last,
                    start_step=last,
                    optimizer=opt,
                    device=device,
                    shuffle_seed=seed + 1000 * r + last,
                    progress_desc=f"phase_a_c{r}_extend",
                    show_progress=True,
                    wandb_log_every=None,
                    collate_fn=collate_fn,
                )
                src = ensure_dir(outs[r] / f"ckpt_{k_star}")
                model.save_pretrained(src)
                tokenizer.save_pretrained(src)
                del model, opt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        dst = ensure_dir(outs[r] / "checkpoint")
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        # Drop intermediate ckpts to save disk.
        for p in outs[r].glob("ckpt_*"):
            if p.is_dir():
                shutil.rmtree(p)
        write_json(
            outs[r] / "phase_a_meta.json",
            {
                **joint_meta,
                "complementary": r,
                "final_step": k_star,
                "earliest_pass": earliest_pass[r],
                "gate_history": gate_history[r],
                "checkpoint": str(dst),
                "target_reached": True,
            },
        )

    joint_meta.update(
        {
            "final_step": k_star,
            "target_reached": True,
            "earliest_pass": {str(k): v for k, v in earliest_pass.items()},
            "gate_history": {str(k): v for k, v in gate_history.items()},
            "checkpoints": {str(r): str(outs[r] / "checkpoint") for r in runs},
        }
    )
    write_json(seed_root / "phase_a_joint_meta.json", joint_meta)
    return joint_meta
