#!/usr/bin/env python3
"""Generate a screened fresh-fact pool (never seen by any Phase-A history).

Excludes panel entities, answer-exposure entities, and reconstructed Phase-B
stream entities. Screens against the base model so the denominator for
"fraction of naive acquisition" is not contaminated by pretraining knowledge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from facts.generate import FactGenConfig, generate_candidates, select_balanced_panel
from facts.screen import screen_facts
from schema import Fact
from util.config import load_yaml
from util.metrics import read_json, write_json
from util.paths import assignment_path, ensure_dir, repo_root, resolve_path
from util.seeding import seed_everything


def _collect_excluded_entities(cfg: dict, seeds: list[int]) -> set[str]:
    facts_df = pd.read_parquet(resolve_path(cfg["paths"]["facts_parquet"]))
    excluded: set[str] = set(facts_df["entity"].astype(str))
    asg_dir = cfg["paths"]["assignments_dir"]
    runs = []
    for r in range(8):
        p = assignment_path(asg_dir, seeds[0], r)
        if p.exists():
            runs.append(r)
    if not runs:
        runs = [0, 1]
    for seed in seeds:
        for r in runs:
            path = assignment_path(asg_dir, seed, r)
            if not path.exists():
                continue
            asg = read_json(path)
            for payload in asg.values():
                ent = payload.get("answer_exposure_entity")
                if ent:
                    excluded.add(str(ent))
    # Reconstruct Phase-B stream entities. Use a fixed base exclude so each
    # stream_seed draw stays O(n) rather than rejection-sampling against a
    # growing set of previous streams.
    base_exclude = set(excluded)
    stream_entities: set[str] = set()
    for seed in seeds:
        stream_seed = int(cfg.get("seed", 42)) + 200 + seed
        stream_seed_dense = stream_seed + 1000 * (abs(hash("dense_early")) % 97)
        for sseed in (stream_seed, stream_seed_dense):
            cand_cfg = FactGenConfig(
                n_relations=6,
                n_answers_per_relation=8,
                n_entities_per_pair=max(1, 3000 // (6 * 8) + 1),
                candidate_buffer=1.0,
                seed=sseed,
            )
            cands = generate_candidates(cand_cfg, exclude_entities=base_exclude)[:3000]
            stream_entities |= {f.entity for f in cands}
    excluded |= stream_entities
    return excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/phase1.yaml")
    parser.add_argument("--n-facts", type=int, default=192)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)) + 777)
    out_parquet = resolve_path("data/facts/fresh_pool.parquet")
    out_jsonl = resolve_path("data/facts/fresh_pool.jsonl")
    summary_path = resolve_path(cfg["paths"]["manifests_dir"]) / "fresh_pool_summary.json"
    if out_parquet.exists() and summary_path.exists() and not args.force:
        print(f"Skip (exists): {out_parquet}")
        return

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    print("Collecting excluded entities (panel + exposure + Phase-B streams)...")
    excluded = _collect_excluded_entities(cfg, seeds)
    print(f"Excluded {len(excluded)} entities")

    n_per_cell = args.n_facts // (6 * 8)
    gen_cfg = FactGenConfig(
        n_relations=6,
        n_answers_per_relation=8,
        n_entities_per_pair=n_per_cell,
        candidate_buffer=1.5,
        seed=int(cfg.get("seed", 42)) + 4242,
    )
    candidates = generate_candidates(gen_cfg, exclude_entities=excluded)
    print(f"Generated {len(candidates)} fresh candidates")

    model = tokenizer = None
    if not args.dry_run:
        model_cfg = load_yaml(cfg["model_config"])
        from train.trainer import load_model_and_tokenizer

        model, tokenizer = load_model_and_tokenizer(
            model_cfg["model_id"],
            torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
        )

    result = screen_facts(
        candidates,
        model=model,
        tokenizer=tokenizer,
        dry_run=args.dry_run or model is None,
        n_entities_per_pair=n_per_cell,
    )
    kept = result.kept
    if len(kept) < args.n_facts:
        raise SystemExit(
            f"Screening left only {len(kept)} facts; need {args.n_facts}. "
            "Increase candidate_buffer or relax screening."
        )
    kept = select_balanced_panel(kept, n_entities_per_pair=n_per_cell)

    ensure_dir(out_parquet.parent)
    rows = [f.to_dict() for f in kept]
    pd.DataFrame(rows).to_parquet(out_parquet, index=False)
    with open(out_jsonl, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary = {
        "n_facts": len(kept),
        "n_candidates": len(candidates),
        "n_rejected": int(result.summary.get("n_rejected", 0)),
        "n_excluded_entities": len(excluded),
        "n_entities_per_pair": n_per_cell,
        "parquet": str(out_parquet.relative_to(repo_root())),
        "screening": result.summary,
    }
    write_json(summary_path, summary)
    print(f"Wrote {out_parquet} ({len(kept)} facts)")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
