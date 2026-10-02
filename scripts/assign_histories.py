#!/usr/bin/env python3
"""Assign histories per optimization seed (2-arm complementary or 3-arm Latin square)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from tqdm import tqdm

from facts.assign import (
    assign_complementary_histories,
    assign_three_arm_histories,
    exposure_audit,
)
from facts.templates import load_templates, write_templates
from schema import Fact
from util.config import load_yaml
from util.metrics import write_json
from util.paths import ensure_dir, resolve_path
from util.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/phase1.yaml")
    parser.add_argument(
        "--mode",
        choices=["complementary", "three_arm"],
        default=None,
        help="Override assignment mode (default: from config.assignment_mode or complementary)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    mode = args.mode or str(cfg.get("assignment_mode", "complementary"))
    out_dir = ensure_dir(cfg["paths"]["assignments_dir"])
    marker = out_dir / "seed_0_c0.json"
    audit_name = (
        "exposure_audit_3arm.json" if mode == "three_arm" else "exposure_audit.json"
    )
    audit_path = resolve_path(cfg["paths"]["manifests_dir"]) / audit_name
    if marker.exists() and audit_path.exists() and not args.force:
        print(f"Skip (exists): {marker}")
        return

    df = pd.read_parquet(resolve_path(cfg["paths"]["facts_parquet"]))
    facts = [Fact.from_dict(r) for r in df.to_dict(orient="records")]
    if mode == "three_arm":
        run_maps = assign_three_arm_histories(
            facts,
            n_seeds=int(cfg.get("n_seeds", 3)),
            seed=int(cfg.get("seed", 42)),
        )
    else:
        run_maps = assign_complementary_histories(
            facts,
            n_seeds=int(cfg.get("n_seeds", 3)),
            seed=int(cfg.get("seed", 42)),
        )
    for (opt_seed, comp), mapping in tqdm(run_maps.items(), desc="write assignments"):
        write_json(out_dir / f"seed_{opt_seed}_c{comp}.json", mapping)

    templates_dir = resolve_path(cfg["paths"]["templates_dir"])
    if not (templates_dir / "phase_a.yaml").exists():
        write_templates(str(templates_dir))
    templates = load_templates(str(templates_dir))
    audit = exposure_audit(facts, run_maps, templates["phase_a"])
    audit["assignment_mode"] = mode
    write_json(audit_path, audit)
    print(f"Wrote assignments ({mode}) -> {out_dir}")
    print(f"Wrote exposure audit -> {audit_path}")


if __name__ == "__main__":
    main()
