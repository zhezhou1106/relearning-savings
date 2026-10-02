#!/usr/bin/env python3
"""Screen candidate facts and write the balanced 576-fact panel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from facts.screen import screen_facts
from schema import Fact
from util.config import load_yaml
from util.metrics import write_json, write_jsonl
from util.paths import ensure_dir, resolve_path
from util.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/phase1.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Skip base-model screen")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    out_parquet = resolve_path(cfg["paths"]["facts_parquet"])
    if out_parquet.exists() and not args.force:
        print(f"Skip (exists): {out_parquet}")
        return

    cand_path = (
        resolve_path(cfg["paths"]["artifacts_dir"])
        / "facts"
        / "candidates"
        / "candidates.parquet"
    )
    df = pd.read_parquet(cand_path)
    facts = [Fact.from_dict(r) for r in df.to_dict(orient="records")]

    model = tokenizer = None
    if not args.dry_run:
        model_cfg = load_yaml(cfg["model_config"])
        from train.trainer import load_model_and_tokenizer

        model, tokenizer = load_model_and_tokenizer(
            model_cfg["model_id"], torch_dtype=model_cfg.get("torch_dtype", "bfloat16")
        )

    result = screen_facts(
        facts,
        model=model,
        tokenizer=tokenizer,
        dry_run=args.dry_run or model is None,
        n_entities_per_pair=int(cfg["facts"]["n_entities_per_pair"]),
    )
    kept = result.kept
    rows = [f.to_dict() for f in kept]
    ensure_dir(out_parquet.parent)
    pd.DataFrame(rows).to_parquet(out_parquet, index=False)
    write_jsonl(resolve_path(cfg["paths"]["facts_jsonl"]), rows)
    summary = dict(result.summary)
    summary["n_kept_panel"] = len(rows)
    summary["target_n_facts"] = int(cfg["facts"]["target_n_facts"])
    write_json(
        resolve_path(cfg["paths"]["manifests_dir"]) / "screening_summary.json",
        summary,
    )
    write_json(
        resolve_path(cfg["paths"]["artifacts_dir"]) / "facts" / "rejected.json",
        result.rejected,
    )
    print(f"Kept {len(rows)} facts -> {out_parquet}")


if __name__ == "__main__":
    main()
