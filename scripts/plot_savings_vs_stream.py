#!/usr/bin/env python3
"""Pooled P5 figure: savings and forgetting versus Phase-B stream step."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from analysis.figures import plot_savings_vs_stream_step
from util.config import load_yaml
from util.metrics import read_json
from util.paths import ensure_dir, resolve_path


def _payload_frame(payload: dict) -> pd.DataFrame:
    return pd.DataFrame(payload.get("checkpoints") or [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/phase1.yaml")
    parser.add_argument("--tag", default="dense_early")
    parser.add_argument("--split", choices=["development", "confirmatory"], default="confirmatory")
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = ensure_dir(resolve_path(cfg["paths"]["artifacts_dir"]) / "analysis" / "p5")
    frames: dict[int, pd.DataFrame] = {}
    for raw in args.seeds.split(","):
        seed = int(raw.strip())
        path = out_dir / f"savings_vs_forgetting_seed_{seed}_{args.tag}_{args.split}.json"
        if not path.exists():
            raise SystemExit(f"Missing analysis JSON: {path}")
        frames[seed] = _payload_frame(read_json(path))

    full = plot_savings_vs_stream_step(
        frames,
        out_path=out_dir / f"savings_vs_stream_step_{args.tag}_{args.split}.pdf",
    )
    zoom = plot_savings_vs_stream_step(
        frames,
        out_path=out_dir / f"savings_vs_stream_step_{args.tag}_{args.split}_trough.pdf",
        drop_early=True,
    )
    print(f"Wrote {full}")
    print(f"Wrote {zoom}")


if __name__ == "__main__":
    main()
