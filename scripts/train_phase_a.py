#!/usr/bin/env python3
"""Phase A for one optimization seed (2-arm pair or 3-arm Latin square)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from facts.templates import load_templates
from schema import Fact
from train.phase_a import run_phase_a_group, run_phase_a_pair
from train.trainer import TrainConfig
from util.config import load_yaml
from util.metrics import read_json
from util.paths import assignment_path, complementary_dir, resolve_path
from util.seeding import seed_everything
from util.wandb_log import wandb_run, wandb_summary


def _discover_runs(cfg: dict, seed: int) -> list[int]:
    runs = []
    for r in range(8):
        if assignment_path(cfg["paths"]["assignments_dir"], seed, r).exists():
            runs.append(r)
    return runs or [0, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/phase1.yaml")
    parser.add_argument("--seed", type=int, required=True, help="Optimization seed index")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)) + args.seed)
    exp = str(cfg["experiment_name"])
    runs = _discover_runs(cfg, args.seed)
    joint_meta = complementary_dir(exp, args.seed, 0).parent / "phase_a_joint_meta.json"
    if joint_meta.exists() and not args.force:
        print(f"Skip (exists): {joint_meta}")
        return

    model_cfg = load_yaml(cfg["model_config"])
    train_cfg_raw = load_yaml(cfg["phase_a_config"])
    opt = train_cfg_raw["optimizer"]
    tr = train_cfg_raw["train"]
    train_cfg = TrainConfig(
        lr=float(opt["lr"]),
        betas=tuple(opt["betas"]),
        weight_decay=float(opt["weight_decay"]),
        grad_clip=float(opt["grad_clip"]),
        effective_batch_size=int(tr["effective_batch_size"]),
        micro_batch_size=int(tr.get("micro_batch_size", 16)),
        max_seq_length=int(tr["max_seq_length"]),
        warmup_updates=int(tr["warmup_updates"]),
        lr_schedule=str(tr["lr_schedule"]),
    )

    facts_df = pd.read_parquet(resolve_path(cfg["paths"]["facts_parquet"]))
    facts = [Fact.from_dict(r) for r in facts_df.to_dict(orient="records")]
    assignments_by_run = {
        r: read_json(assignment_path(cfg["paths"]["assignments_dir"], args.seed, r))
        for r in runs
    }
    all_templates = load_templates(cfg["paths"]["templates_dir"])
    splits = read_json(resolve_path(cfg["paths"]["splits"]))
    gate_fact_ids = list(splits["development"])

    with wandb_run(
        name=f"{exp}-seed{args.seed}-phase_a",
        job_type="phase_a",
        group=f"{exp}-seed{args.seed}",
        tags=[exp, f"seed_{args.seed}", "phase_a", f"n_runs_{len(runs)}"],
        config={
            "experiment": exp,
            "seed": args.seed,
            "phase": "phase_a",
            "n_runs": len(runs),
            "dry_run": args.dry_run,
        },
        dir=str(resolve_path(cfg["paths"]["artifacts_dir"]) / "wandb"),
    ):
        if len(runs) == 2:
            meta = run_phase_a_pair(
                facts=facts,
                assignments_c0=assignments_by_run[0],
                assignments_c1=assignments_by_run[1],
                phase_a_templates=all_templates["phase_a"],
                behavioral_templates=all_templates["behavioral"],
                model_id=model_cfg["model_id"],
                out_dir_c0=complementary_dir(exp, args.seed, 0) / "phase_a",
                out_dir_c1=complementary_dir(exp, args.seed, 1) / "phase_a",
                train_cfg=train_cfg,
                torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
                max_updates=int(tr.get("max_updates", 8000)),
                dry_run=args.dry_run,
                target_exact_match=float(tr.get("target_exact_match", 0.90)),
                min_log_odds_separation=float(tr.get("min_log_odds_separation", 1.0)),
                control_residual_log_odds_max=float(
                    tr.get("control_residual_log_odds_max", -0.5)
                ),
                eval_every_updates=int(tr.get("eval_every_updates", 250)),
                eval_facts_per_history=int(tr.get("eval_facts_per_history", 96)),
                require_target=bool(tr.get("require_target_exact_match", True)),
                seed=int(cfg.get("seed", 42)) + args.seed,
                gate_fact_ids=gate_fact_ids,
                answer_token_only_loss=bool(tr.get("answer_token_only_loss", True)),
            )
        else:
            meta = run_phase_a_group(
                facts=facts,
                assignments_by_run=assignments_by_run,
                phase_a_templates=all_templates["phase_a"],
                behavioral_templates=all_templates["behavioral"],
                model_id=model_cfg["model_id"],
                out_dirs={
                    r: complementary_dir(exp, args.seed, r) / "phase_a" for r in runs
                },
                train_cfg=train_cfg,
                torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
                max_updates=int(tr.get("max_updates", 8000)),
                dry_run=args.dry_run,
                target_exact_match=float(tr.get("target_exact_match", 0.90)),
                min_log_odds_separation=float(tr.get("min_log_odds_separation", 1.0)),
                control_residual_log_odds_max=float(
                    tr.get("control_residual_log_odds_max", -0.5)
                ),
                wrong_residual_log_odds_max=float(
                    tr.get("wrong_residual_log_odds_max", -1.0)
                ),
                eval_every_updates=int(tr.get("eval_every_updates", 250)),
                eval_facts_per_history=int(tr.get("eval_facts_per_history", 64)),
                require_target=bool(tr.get("require_target_exact_match", True)),
                seed=int(cfg.get("seed", 42)) + args.seed,
                gate_fact_ids=gate_fact_ids,
                answer_token_only_loss=bool(tr.get("answer_token_only_loss", True)),
            )
        wandb_summary(
            {
                "final_step": meta.get("final_step"),
                "target_reached": meta.get("target_reached"),
                "n_runs": len(runs),
            }
        )
    print(f"Phase A done -> seed_{args.seed} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
