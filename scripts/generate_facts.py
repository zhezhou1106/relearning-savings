#!/usr/bin/env python3
"""Generate synthetic candidate facts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from facts.generate import FactGenConfig, facts_to_rows, generate_candidates
from util.config import load_yaml
from util.metrics import write_json, write_jsonl
from util.paths import ensure_dir, resolve_path
from util.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/phase1.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    out_parquet = (
        resolve_path(cfg["paths"]["artifacts_dir"])
        / "facts"
        / "candidates"
        / "candidates.parquet"
    )
    out_jsonl = (
        resolve_path(cfg["paths"]["artifacts_dir"])
        / "facts"
        / "candidates"
        / "candidates.jsonl"
    )
    if out_parquet.exists() and not args.force:
        print(f"Skip (exists): {out_parquet}")
        return

    fcfg = FactGenConfig(
        n_relations=int(cfg["facts"]["n_relations"]),
        n_answers_per_relation=int(cfg["facts"]["n_answers_per_relation"]),
        n_entities_per_pair=int(cfg["facts"]["n_entities_per_pair"]),
        candidate_buffer=float(cfg["facts"]["candidate_buffer"]),
        seed=int(cfg.get("seed", 42)),
    )
    facts = generate_candidates(fcfg)
    rows = facts_to_rows(facts)
    ensure_dir(out_parquet.parent)
    pd.DataFrame(rows).to_parquet(out_parquet, index=False)
    write_jsonl(out_jsonl, rows)
    write_json(
        resolve_path(cfg["paths"]["manifests_dir"]) / "generate_facts.json",
        {"n_candidates": len(rows), "config": fcfg.__dict__},
    )
    print(f"Wrote {len(rows)} candidates -> {out_parquet}")


if __name__ == "__main__":
    main()
