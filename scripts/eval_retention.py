#!/usr/bin/env python3
"""Behavioral retention at Phase-B checkpoints for one complementary trajectory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from tqdm import tqdm

from data.datasets import relation_answer_list
from eval.behavioral import answer_token_logprob, score_fact
from facts.templates import load_templates, render
from schema import Fact, History
from util.config import load_yaml
from util.metrics import read_json, write_json
from util.paths import (
    assignment_path,
    complementary_dir,
    ensure_dir,
    eval_dirname,
    phase_b_dirname,
    resolve_path,
)
from util.wandb_log import wandb_log, wandb_run, wandb_summary


def _per_template_probs(
    model,
    tokenizer,
    fact: Fact,
    templates: list[str],
    *,
    device: str,
) -> list[list[float]]:
    """8-way probability vector for each behavioral template (not averaged)."""
    answers = relation_answer_list(fact.relation)
    out: list[list[float]] = []
    for template in templates:
        prompt = render(template, fact)
        if prompt.endswith(fact.answer):
            prompt = prompt[: -len(fact.answer)].rstrip()
        logp = np.asarray(
            [
                answer_token_logprob(model, tokenizer, prompt, ans, device=device)
                for ans in answers
            ],
            dtype=np.float64,
        )
        logp = logp - np.max(logp)
        probs = np.exp(logp)
        probs = probs / probs.sum()
        out.append([float(x) for x in probs])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/phase1.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--complementary",
        type=int,
        required=True,
        help="Run index (0/1 for two-arm; 0/1/2 for three-arm)",
    )
    parser.add_argument("--phase-b-config", default=None)
    parser.add_argument("--tag", default=None, help="Read phase_b_<tag>, write eval_<tag>")
    parser.add_argument("--steps", type=int, nargs="*", default=None)
    parser.add_argument(
        "--per-template",
        action="store_true",
        help="Also persist per-template answer_probs (for measurement-noise JS floor)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    exp = str(cfg["experiment_name"])
    run_root = complementary_dir(exp, args.seed, args.complementary)
    phase_b_cfg = load_yaml(args.phase_b_config or cfg["phase_b_config"])
    steps = args.steps or list(phase_b_cfg["train"]["checkpoint_steps"])
    facts_df = pd.read_parquet(resolve_path(cfg["paths"]["facts_parquet"]))
    facts = [Fact.from_dict(r) for r in facts_df.to_dict(orient="records")]
    assignments = read_json(
        assignment_path(cfg["paths"]["assignments_dir"], args.seed, args.complementary)
    )
    templates = load_templates(cfg["paths"]["templates_dir"])["behavioral"]
    model_dtype = str(load_yaml(cfg["model_config"]).get("torch_dtype", "bfloat16"))
    splits = read_json(resolve_path(cfg["paths"]["splits"]))
    eval_ids = set(splits["development"]) | set(splits["confirmatory"])
    facts = [f for f in facts if f.fact_id in eval_ids]

    out_dir = ensure_dir(run_root / eval_dirname(args.tag))
    pb_dir = run_root / phase_b_dirname(args.tag)
    tag_label = args.tag or "baseline"
    with wandb_run(
        name=f"{exp}-seed{args.seed}-c{args.complementary}-eval-{tag_label}",
        job_type="eval_retention",
        group=f"{exp}-seed{args.seed}",
        tags=[exp, f"seed_{args.seed}", f"c{args.complementary}", "eval", tag_label],
        config={
            "seed": args.seed,
            "complementary": args.complementary,
            "steps": steps,
            "tag": args.tag,
            "per_template": args.per_template,
        },
        dir=str(resolve_path(cfg["paths"]["artifacts_dir"]) / "wandb"),
    ):
        for step in tqdm(steps, desc="eval retention"):
            out_path = out_dir / f"behavioral_{step}.parquet"
            rows = []
            existing = None
            if args.per_template and out_path.exists() and not args.force:
                existing = pd.read_parquet(out_path)
                if "answer_probs_per_template" in existing.columns:
                    print(f"Skip (per-template exists): {out_path}")
                    continue
            elif out_path.exists() and not args.force:
                print(f"Skip (exists): {out_path}")
                continue
            if args.dry_run:
                for fact in facts:
                    hist: History = assignments[fact.fact_id]["history"]
                    row = {
                        "fact_id": fact.fact_id,
                        "history": hist,
                        "stream_steps": step,
                        "exact_match": 0.0,
                        "target_log_odds": 0.0,
                        "target_rank": 8,
                        "answer_probs": [0.125] * 8,
                        "split": (
                            "development"
                            if fact.fact_id in splits["development"]
                            else "confirmatory"
                        ),
                    }
                    if args.per_template:
                        row["answer_probs_per_template"] = [[0.125] * 8] * 2
                    rows.append(row)
            else:
                import torch

                from train.trainer import load_model_and_tokenizer

                ckpt = pb_dir / f"ckpt_{step}"
                if not ckpt.exists():
                    print(f"Skip (missing ckpt): {ckpt}")
                    continue
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model, tokenizer = load_model_and_tokenizer(
                    str(ckpt), torch_dtype=model_dtype
                )
                model.to(device)
                model.eval()
                # If only filling per-template on existing scores, reuse averaged metrics.
                existing_by_id = {}
                if existing is not None:
                    existing_by_id = {
                        str(r.fact_id): r for r in existing.itertuples(index=False)
                    }
                for fact in tqdm(facts, desc=f"score ckpt {step}", leave=False):
                    hist = assignments[fact.fact_id]["history"]
                    if fact.fact_id in existing_by_id and args.per_template:
                        er = existing_by_id[fact.fact_id]
                        row = {
                            "fact_id": fact.fact_id,
                            "history": hist,
                            "stream_steps": step,
                            "exact_match": float(er.exact_match),
                            "target_log_odds": float(er.target_log_odds),
                            "target_rank": int(er.target_rank),
                            "answer_probs": list(er.answer_probs),
                            "split": str(er.split),
                            "answer_probs_per_template": _per_template_probs(
                                model,
                                tokenizer,
                                fact,
                                templates[fact.relation],
                                device=device,
                            ),
                        }
                    else:
                        scores = score_fact(
                            model,
                            tokenizer,
                            fact,
                            templates[fact.relation],
                            history=hist,
                            device=device,
                        )
                        row = {
                            "fact_id": scores.fact_id,
                            "history": scores.history,
                            "stream_steps": step,
                            "exact_match": scores.exact_match,
                            "target_log_odds": scores.target_log_odds,
                            "target_rank": scores.target_rank,
                            "answer_probs": list(scores.answer_probs),
                            "split": (
                                "development"
                                if fact.fact_id in splits["development"]
                                else "confirmatory"
                            ),
                        }
                        if args.per_template:
                            row["answer_probs_per_template"] = _per_template_probs(
                                model,
                                tokenizer,
                                fact,
                                templates[fact.relation],
                                device=device,
                            )
                    rows.append(row)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            pd.DataFrame(rows).to_parquet(out_path, index=False)
            write_json(
                out_dir / f"behavioral_{step}_meta.json",
                {
                    "n_facts": len(rows),
                    "stream_steps": step,
                    "tag": args.tag,
                    "per_template": args.per_template,
                },
            )
            wandb_log({"eval/n_facts": len(rows)}, step=step)
            print(f"Wrote {out_path}")
        wandb_summary({"n_checkpoints": len(steps)})


if __name__ == "__main__":
    main()
