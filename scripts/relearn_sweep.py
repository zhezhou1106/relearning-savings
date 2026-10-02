#!/usr/bin/env python3
"""P5 relearning (k in eval_ks) at multiple Phase B stream checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from tqdm import tqdm

from facts.templates import load_templates
from schema import Fact
from train.relearn import load_relearn_checkpoint, relearn_fact_curve
from train.trainer import TrainConfig
from util.config import load_yaml
from util.metrics import read_json
from util.paths import (
    assignment_path,
    complementary_dir,
    ensure_dir,
    phase_b_dirname,
    relearn_merged_path,
    relearn_stream_dir,
    resolve_path,
)
from util.seeding import seed_everything
from util.wandb_log import wandb_log, wandb_run, wandb_summary


def _parse_steps(raw: str) -> list[int]:
    steps = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not steps:
        raise SystemExit("No stream steps provided")
    return steps


def _fact_ids(cfg: dict, split: str) -> list[str]:
    splits = read_json(resolve_path(cfg["paths"]["splits"]))
    if split == "development":
        return list(splits["development"])
    if split == "confirmatory":
        return list(splits["confirmatory"])
    raise SystemExit(f"Unknown split: {split}")


def relearn_at_step(
    *,
    cfg: dict,
    seed: int,
    complementary: int,
    tag: str,
    stream_steps: int,
    fact_ids: list[str],
    dry_run: bool,
    force: bool,
    max_facts: int | None,
    split: str = "development",
    phase_b_tag: str | None = None,
    relearn_tag: str | None = None,
    relearn_config: str | None = None,
    fact_source: str = "split",
    fresh_pool_path: str | None = None,
) -> Path:
    model_cfg = load_yaml(cfg["model_config"])
    relearn_cfg = load_yaml(relearn_config or cfg["relearn_config"])
    opt = relearn_cfg["optimizer"]
    tr = relearn_cfg["train"]
    train_cfg = TrainConfig(
        lr=float(opt["lr"]),
        betas=tuple(opt["betas"]),
        weight_decay=float(opt["weight_decay"]),
        grad_clip=float(opt["grad_clip"]),
        effective_batch_size=int(tr.get("effective_batch_size", 2)),
        micro_batch_size=int(tr.get("micro_batch_size", 2)),
        max_seq_length=int(tr["max_seq_length"]),
        warmup_updates=int(tr.get("warmup_updates", 0)),
        lr_schedule=str(tr.get("lr_schedule", "constant")),
    )
    eval_ks = list(tr["eval_ks"])
    templates = load_templates(cfg["paths"]["templates_dir"])
    if max_facts is not None:
        fact_ids = fact_ids[:max_facts]

    if fact_source == "fresh_pool":
        pool_path = resolve_path(fresh_pool_path or "data/facts/fresh_pool.parquet")
        facts_df = pd.read_parquet(pool_path)
        facts = {
            Fact.from_dict(r).fact_id: Fact.from_dict(r)
            for r in facts_df.to_dict(orient="records")
        }
        fact_ids = list(facts.keys()) if not fact_ids else [f for f in fact_ids if f in facts]
        if max_facts is not None:
            fact_ids = fact_ids[:max_facts]
        assignments: dict = {}
        history_default = "fresh"
        split_label = "fresh_pool"
    else:
        facts_df = pd.read_parquet(resolve_path(cfg["paths"]["facts_parquet"]))
        facts = {
            Fact.from_dict(r).fact_id: Fact.from_dict(r)
            for r in facts_df.to_dict(orient="records")
        }
        assignments = read_json(
            assignment_path(cfg["paths"]["assignments_dir"], seed, complementary)
        )
        history_default = None
        split_label = split

    exp = str(cfg["experiment_name"])
    ckpt_tag = phase_b_tag or tag
    out_tag = relearn_tag or tag
    run_root = complementary_dir(exp, seed, complementary)
    ckpt = run_root / phase_b_dirname(ckpt_tag) / f"ckpt_{stream_steps}"
    if not ckpt.exists() and not dry_run:
        raise SystemExit(f"Missing checkpoint: {ckpt}")

    out_root = ensure_dir(
        relearn_stream_dir(exp, seed, complementary, out_tag, stream_steps, split_label)
    )
    curves_path = out_root / "curves.parquet"
    if curves_path.exists() and not force:
        print(f"Skip (exists): {curves_path}")
        return curves_path

    role_rows: list[dict] = []
    with wandb_run(
        name=f"{exp}-seed{seed}-c{complementary}-relearn-{out_tag}-k{stream_steps}",
        job_type="relearn_sweep",
        group=f"{exp}-seed{seed}-{out_tag}",
        tags=[exp, f"seed_{seed}", f"c{complementary}", out_tag, split_label, "relearn_sweep"],
        config={
            "seed": seed,
            "complementary": complementary,
            "tag": out_tag,
            "phase_b_tag": ckpt_tag,
            "relearn_config": relearn_config or cfg["relearn_config"],
            "split": split_label,
            "fact_source": fact_source,
            "stream_steps": stream_steps,
            "n_facts": len(fact_ids),
            "eval_ks": eval_ks,
            "lr": float(opt["lr"]),
            "dry_run": dry_run,
        },
        dir=str(resolve_path(cfg["paths"]["artifacts_dir"]) / "wandb"),
    ):
        model = tokenizer = base_state = device = None
        if not dry_run:
            model, tokenizer, base_state, device = load_relearn_checkpoint(
                ckpt, torch_dtype=str(model_cfg.get("torch_dtype", "bfloat16"))
            )
        try:
            for i, fid in enumerate(tqdm(fact_ids, desc=f"relearn@{stream_steps}")):
                fact = facts[fid]
                if history_default is not None:
                    hist = history_default
                else:
                    hist = assignments[fid]["history"]
                curve = relearn_fact_curve(
                    fact=fact,
                    ckpt_path=ckpt,
                    train_templates=templates["relearn_train"][fact.relation],
                    eval_templates=templates["relearn_eval"][fact.relation],
                    eval_ks=eval_ks,
                    train_cfg=train_cfg,
                    out_dir=out_root / fid,
                    dry_run=dry_run,
                    model=model,
                    tokenizer=tokenizer,
                    base_state=base_state,
                    device=device,
                    n_train_templates=int(tr.get("n_train_templates", 2)),
                    n_eval_templates=int(tr.get("n_eval_templates", 2)),
                    target_prob_threshold=float(tr.get("target_prob_threshold", 0.90)),
                    answer_token_only_loss=bool(tr.get("answer_token_only_loss", True)),
                )
                for row in curve:
                    row.update(
                        {
                            "history": hist,
                            "checkpoint_role": "forgetting_sweep",
                            "phase_b_tag": ckpt_tag,
                            "relearn_tag": out_tag,
                            "split": split_label,
                            "stream_steps": stream_steps,
                            "seed": seed,
                            "complementary": complementary,
                        }
                    )
                    role_rows.append(row)
                if (i + 1) % 16 == 0:
                    wandb_log({"relearn/facts_done": i + 1}, step=i + 1)
        finally:
            if model is not None:
                del model, base_state

        pd.DataFrame(role_rows).to_parquet(curves_path, index=False)
        wandb_summary({"n_curves": len(fact_ids), "stream_steps": stream_steps})
        print(f"Wrote {curves_path}")
        return curves_path


def merge_curves(
    cfg: dict,
    seed: int,
    tag: str,
    steps: list[int],
    split: str = "development",
    runs: list[int] | None = None,
) -> Path:
    exp = str(cfg["experiment_name"])
    if runs is None:
        seed_root = complementary_dir(exp, seed, 0).parent
        runs = []
        for p in sorted(seed_root.glob("c*")):
            if p.is_dir() and p.name.startswith("c") and p.name[1:].isdigit():
                runs.append(int(p.name[1:]))
        runs = runs or [0, 1]
    parts: list[pd.DataFrame] = []
    for comp in runs:
        for stream_steps in steps:
            path = (
                relearn_stream_dir(exp, seed, comp, tag, stream_steps, split)
                / "curves.parquet"
            )
            if path.exists():
                parts.append(pd.read_parquet(path))
    if not parts:
        raise SystemExit(f"No curves found for tag={tag} seed={seed} split={split}")
    merged = pd.concat(parts, ignore_index=True)
    out = ensure_dir(relearn_merged_path(exp, seed, tag, split).parent) / "all_curves.parquet"
    merged.to_parquet(out, index=False)
    print(f"Wrote merged curves ({len(merged)} rows) -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/phase1.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--complementary", type=int, required=True)
    parser.add_argument("--tag", default="dense_early")
    parser.add_argument(
        "--phase-b-tag",
        default=None,
        help="Phase B checkpoint tag (default: --tag)",
    )
    parser.add_argument(
        "--relearn-tag",
        default=None,
        help="Output relearn directory tag (default: --tag)",
    )
    parser.add_argument(
        "--relearn-config",
        default=None,
        help="Override relearn yaml (default: experiment relearn_config)",
    )
    parser.add_argument(
        "--stream-steps",
        required=True,
        help="Comma-separated Phase B update counts (e.g. 0,25,50,...)",
    )
    parser.add_argument(
        "--split",
        choices=["development", "confirmatory"],
        default="development",
        help="Fact panel; development matches dense_early behavioral timeline",
    )
    parser.add_argument(
        "--fact-source",
        choices=["split", "fresh_pool"],
        default="split",
        help="split: use development/confirmatory panel; fresh_pool: naive-acquisition arm",
    )
    parser.add_argument(
        "--fresh-pool",
        default="data/facts/fresh_pool.parquet",
        help="Parquet for --fact-source fresh_pool",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-facts", type=int, default=None)
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge per-checkpoint curves into relearn/<tag>/all_curves.parquet",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    steps = _parse_steps(args.stream_steps)
    relearn_tag = args.relearn_tag or args.tag

    if args.merge_only:
        merge_curves(cfg, args.seed, relearn_tag, steps, split=args.split)
        return

    seed_everything(int(cfg.get("seed", 42)) + 300 + args.seed + 10 * args.complementary)
    if args.fact_source == "fresh_pool":
        fact_ids = []  # filled inside relearn_at_step from the pool parquet
    else:
        fact_ids = _fact_ids(cfg, args.split)
    for stream_steps in steps:
        relearn_at_step(
            cfg=cfg,
            seed=args.seed,
            complementary=args.complementary,
            tag=args.tag,
            stream_steps=stream_steps,
            fact_ids=fact_ids,
            dry_run=args.dry_run,
            force=args.force,
            max_facts=args.max_facts,
            split=args.split,
            phase_b_tag=args.phase_b_tag,
            relearn_tag=relearn_tag,
            relearn_config=args.relearn_config,
            fact_source=args.fact_source,
            fresh_pool_path=args.fresh_pool,
        )


if __name__ == "__main__":
    main()
