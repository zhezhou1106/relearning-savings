#!/usr/bin/env python3
"""Phase B for one complementary trajectory; shared stream seed within a pair."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from eval.stream_health import DEFAULT_FORMAT_PROMPTS, load_general_validation
from facts.templates import load_templates
from schema import Fact
from train.phase_b import run_phase_b
from train.trainer import TrainConfig
from util.config import load_yaml
from util.metrics import read_json
from util.paths import assignment_path, complementary_dir, phase_b_dirname, resolve_path
from util.seeding import seed_everything
from util.wandb_log import wandb_run, wandb_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/phase1.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--complementary", type=int, required=True)
    parser.add_argument(
        "--phase-b-config",
        default=None,
        help="Override phase_b yaml (defaults to config.phase_b_config)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Write under phase_b_<tag> instead of phase_b (pilots)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    # Shared stream RNG within complementary pair (depends on seed only).
    seed_everything(int(cfg.get("seed", 42)) + 100 + args.seed)
    exp = str(cfg["experiment_name"])
    run_root = complementary_dir(exp, args.seed, args.complementary)
    out_dir = run_root / phase_b_dirname(args.tag)
    if (out_dir / "phase_b_meta.json").exists() and not args.force:
        # Resume-friendly: meta written at start; require force only if fully done.
        steps = load_yaml(args.phase_b_config or cfg["phase_b_config"])["train"][
            "checkpoint_steps"
        ]
        last = max(int(s) for s in steps)
        if (out_dir / f"ckpt_{last}" / "config.json").exists() and not args.force:
            print(f"Skip (exists): {out_dir}")
            return

    pb_path = args.phase_b_config or cfg["phase_b_config"]
    train_cfg_raw = load_yaml(pb_path)
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
    all_templates = load_templates(cfg["paths"]["templates_dir"])
    templates = all_templates["phase_a"]
    general_texts = load_general_validation(
        resolve_path(
            cfg["paths"].get("general_validation", "data/eval/general_validation.txt")
        )
    )
    phase_a_ckpt = run_root / "phase_a" / "checkpoint"

    facts_df = pd.read_parquet(resolve_path(cfg["paths"]["facts_parquet"]))
    panel_facts = [Fact.from_dict(r) for r in facts_df.to_dict(orient="records")]
    # Exclude panel + exposure entities from ALL runs of this seed.
    phase_a_entities = set(facts_df["entity"])
    asg_dir = cfg["paths"]["assignments_dir"]
    for r in range(8):
        path = assignment_path(asg_dir, args.seed, r)
        if not path.exists():
            continue
        asg = read_json(path)
        phase_a_entities |= {
            a["answer_exposure_entity"]
            for a in asg.values()
            if a.get("answer_exposure_entity")
        }
    model_cfg = load_yaml(cfg["model_config"])
    stream_seed = int(cfg.get("seed", 42)) + 200 + args.seed
    # Distinct stream seed per pilot tag so overwrite vs dense don't collide.
    if args.tag:
        stream_seed += 1000 * (abs(hash(args.tag)) % 97)

    tag_label = args.tag or "baseline"
    with wandb_run(
        name=f"{exp}-seed{args.seed}-c{args.complementary}-phase_b-{tag_label}",
        job_type="phase_b",
        group=f"{exp}-seed{args.seed}",
        tags=[exp, f"seed_{args.seed}", f"c{args.complementary}", "phase_b", tag_label],
        config={
            "experiment": exp,
            "seed": args.seed,
            "complementary": args.complementary,
            "stream_seed": stream_seed,
            "tag": args.tag,
            "phase_b_config": pb_path,
            "dry_run": args.dry_run,
        },
        dir=str(resolve_path(cfg["paths"]["artifacts_dir"]) / "wandb"),
    ):
        meta = run_phase_b(
            phase_a_ckpt=phase_a_ckpt,
            phase_a_templates=templates,
            out_dir=out_dir,
            train_cfg=train_cfg,
            total_updates=int(tr["total_updates"]),
            n_stream_facts=int(tr["n_stream_facts"]),
            checkpoint_steps=list(tr["checkpoint_steps"]),
            seed=stream_seed,
            dry_run=args.dry_run,
            exclude_entities=phase_a_entities,
            model_dtype=str(model_cfg.get("torch_dtype", "bfloat16")),
            general_texts=general_texts,
            format_prompts=DEFAULT_FORMAT_PROMPTS,
            panel_facts=panel_facts,
            overwrite_panel=bool(tr.get("overwrite_panel", False)),
            overwrite_scope=tr.get("overwrite_scope"),
            overwrite_relation=tr.get("overwrite_relation"),
            answer_token_only_loss=bool(tr.get("answer_token_only_loss", False)),
            n_templates_per_fact=int(tr.get("n_templates_per_fact", 2)),
        )
        wandb_summary(
            {
                "n_stream_examples": meta.get("n_stream_examples"),
                "n_overwrite_examples": meta.get("n_overwrite_examples"),
                "total_updates": meta.get("total_updates"),
            }
        )
    print(f"Phase B done -> seed_{args.seed}/c{args.complementary} tag={args.tag}")


if __name__ == "__main__":
    main()
