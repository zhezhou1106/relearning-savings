#!/usr/bin/env python3
"""Write template banks and freeze development/confirmatory splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from tqdm import tqdm

from facts.assign import partition_splits
from facts.templates import write_templates
from schema import Fact
from util.config import load_yaml
from util.metrics import write_json
from util.paths import resolve_path
from util.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/phase1.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    splits_path = resolve_path(cfg["paths"]["splits"])
    templates_dir = resolve_path(cfg["paths"]["templates_dir"])
    if splits_path.exists() and templates_dir.exists() and not args.force:
        print(f"Skip (exists): {splits_path}")
        return

    with tqdm(total=2, desc="templates + splits") as pbar:
        write_templates(str(templates_dir))
        pbar.update(1)
        pbar.set_postfix_str("partition splits")
        df = pd.read_parquet(resolve_path(cfg["paths"]["facts_parquet"]))
        facts = [Fact.from_dict(r) for r in df.to_dict(orient="records")]
        splits = partition_splits(
            facts,
            development=int(cfg["splits"]["development"]),
            confirmatory=int(cfg["splits"]["confirmatory"]),
            seed=int(cfg.get("seed", 42)),
        )
        pbar.update(1)

    write_json(splits_path, splits)
    write_json(
        resolve_path(cfg["paths"]["manifests_dir"]) / "splits_summary.json",
        {k: len(v) for k, v in splits.items()},
    )
    print(f"Wrote templates -> {templates_dir}")
    print(f"Wrote splits -> {splits_path}")


if __name__ == "__main__":
    main()
