#!/usr/bin/env python3
"""Nominate earliest LO+JS equivalence checkpoint (dev) and confirm on frozen panel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from tqdm import tqdm

from checkpoints.equivalence import BehavioralMargins, behavioral_equivalent, simultaneous_alpha
from checkpoints.select import select_checkpoints
from eval.behavioral import aggregate_paired_fact_gaps
from schema import BehavioralScores
from util.config import load_yaml
from util.metrics import read_json, write_json
from util.paths import (
    complementary_dir,
    ensure_dir,
    eval_dirname,
    resolve_path,
    selected_checkpoints_path,
)


def _scores_from_df(df: pd.DataFrame) -> dict[str, BehavioralScores]:
    out: dict[str, BehavioralScores] = {}
    for r in df.to_dict(orient="records"):
        probs = r.get("answer_probs")
        if probs is None:
            probs = [0.125] * 8
        else:
            probs = list(probs)
        out[str(r["fact_id"])] = BehavioralScores(
            fact_id=str(r["fact_id"]),
            history=r["history"],
            exact_match=float(r["exact_match"]),
            target_log_odds=float(r["target_log_odds"]),
            target_rank=int(r["target_rank"]),
            answer_probs=tuple(float(x) for x in probs),
        )
    return out


def _discover_runs(exp: str, seed: int) -> list[int]:
    seed_root = complementary_dir(exp, seed, 0).parent
    runs = []
    for p in sorted(seed_root.glob("c*")):
        if p.is_dir() and p.name.startswith("c") and p.name[1:].isdigit():
            runs.append(int(p.name[1:]))
    return runs or [0, 1]


def _pair_gaps_at_step(
    *,
    exp: str,
    seed: int,
    step: int,
    fact_ids: set[str],
    answer_ids: dict[str, int],
    n_boot: int,
    alpha: float,
    cfg_seed: int,
    tag: str | None = None,
    treatment: str = "learned",
    reference: str = "control",
    runs: list[int] | None = None,
) -> dict:
    frames = []
    for comp in runs or _discover_runs(exp, seed):
        path = (
            complementary_dir(exp, seed, comp)
            / eval_dirname(tag)
            / f"behavioral_{step}.parquet"
        )
        if not path.exists():
            return {}
        df = pd.read_parquet(path)
        df = df[df["fact_id"].isin(fact_ids)]
        frames.append(df)
    by_arm: dict[str, dict[str, BehavioralScores]] = {}
    for df in frames:
        for fid, score in _scores_from_df(df).items():
            by_arm.setdefault(score.history, {})[fid] = score
    if treatment not in by_arm or reference not in by_arm:
        return {}
    return aggregate_paired_fact_gaps(
        by_arm[treatment],
        by_arm[reference],
        target_answer_ids=answer_ids,
        n_boot=n_boot,
        seed=cfg_seed + step,
        alpha=alpha,
    )


def _all_pairwise_gaps(
    *,
    exp: str,
    seed: int,
    step: int,
    fact_ids: set[str],
    answer_ids: dict[str, int],
    n_boot: int,
    alpha: float,
    cfg_seed: int,
    tag: str | None,
    runs: list[int],
) -> dict:
    """Emit learned-control (primary) plus any other arm pairs present."""
    arms_present: set[str] = set()
    for comp in runs:
        path = (
            complementary_dir(exp, seed, comp)
            / eval_dirname(tag)
            / f"behavioral_{step}.parquet"
        )
        if path.exists():
            df = pd.read_parquet(path, columns=["history"])
            arms_present |= set(df["history"].astype(str))
    pairs = [("learned", "control")]
    if "wrong" in arms_present:
        pairs.extend([("learned", "wrong"), ("wrong", "control")])
    result: dict = {}
    for treat, ref in pairs:
        gaps = _pair_gaps_at_step(
            exp=exp,
            seed=seed,
            step=step,
            fact_ids=fact_ids,
            answer_ids=answer_ids,
            n_boot=n_boot,
            alpha=alpha,
            cfg_seed=cfg_seed + hash((treat, ref)) % 10_000,
            tag=tag,
            treatment=treat,
            reference=ref,
            runs=runs,
        )
        if treat == "learned" and ref == "control":
            result.update(gaps)
        result[f"{treat}_vs_{ref}"] = gaps
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/phase1.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--phase-b-config", default=None)
    parser.add_argument("--behavioral-config", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_path = selected_checkpoints_path(
        cfg["paths"]["manifests_dir"], args.seed, tag=args.tag
    )
    if out_path.exists() and not args.force:
        print(f"Skip (exists): {out_path}")
        return

    beh_cfg = load_yaml(args.behavioral_config or cfg["behavioral_config"])
    margins = BehavioralMargins(
        log_odds=float(beh_cfg["margins"]["log_odds"]),
        js_distance=float(beh_cfg["margins"]["js_distance"]),
    )
    phase_b_cfg = load_yaml(args.phase_b_config or cfg["phase_b_config"])
    steps = list(phase_b_cfg["train"]["checkpoint_steps"])
    alpha = simultaneous_alpha(
        1.0 - float(beh_cfg.get("confidence_level", 0.95)),
        int(beh_cfg.get("n_behavioral_quantities", 2)),
        method=str(beh_cfg.get("simultaneous_correction", "none")),
    )
    n_boot = int(beh_cfg.get("n_boot", 2000))
    splits = read_json(resolve_path(cfg["paths"]["splits"]))
    dev_ids = set(splits["development"])
    conf_ids = set(splits["confirmatory"])

    import pandas as pd

    facts_df = pd.read_parquet(resolve_path(cfg["paths"]["facts_parquet"]))
    answer_ids = {str(r.fact_id): int(r.answer_id) for r in facts_df.itertuples()}
    exp = str(cfg["experiment_name"])
    cfg_seed = int(cfg.get("seed", 42)) + 1000 * args.seed

    timeline = []
    runs = _discover_runs(exp, args.seed)
    for step in tqdm(steps, desc="nominate"):
        gaps = _all_pairwise_gaps(
            exp=exp,
            seed=args.seed,
            step=step,
            fact_ids=dev_ids,
            answer_ids=answer_ids,
            n_boot=n_boot,
            alpha=alpha,
            cfg_seed=cfg_seed,
            tag=args.tag,
            runs=runs,
        )
        row: dict = {"stream_steps": step, "n_runs": len(runs)}
        if gaps and "delta_log_odds" in gaps:
            for k, v in gaps.items():
                row[k] = v
            row["behavioral_equivalent"] = behavioral_equivalent(
                delta_log_odds_ci=tuple(gaps["delta_log_odds_ci"]),
                mean_js_ci=tuple(gaps["mean_js_ci"]),
                margins=margins,
            )
            seed_dir = complementary_dir(exp, args.seed, 0).parent
            write_json(seed_dir / f"dev_gaps_{step}.json", gaps)
        else:
            row["behavioral_equivalent"] = False
        timeline.append(row)

    selected = select_checkpoints(seed=args.seed, timeline=timeline, margins_b=margins)

    confirmatory_pass = None
    confirmatory_gaps = {}
    if selected.equivalence_steps is not None:
        confirmatory_gaps = _pair_gaps_at_step(
            exp=exp,
            seed=args.seed,
            step=selected.equivalence_steps,
            fact_ids=conf_ids,
            answer_ids=answer_ids,
            n_boot=n_boot,
            alpha=alpha,
            cfg_seed=cfg_seed + 7,
            tag=args.tag,
        )
        if confirmatory_gaps:
            confirmatory_pass = behavioral_equivalent(
                delta_log_odds_ci=tuple(confirmatory_gaps["delta_log_odds_ci"]),
                mean_js_ci=tuple(confirmatory_gaps["mean_js_ci"]),
                margins=margins,
            )

    payload = {
        "seed": selected.seed,
        "tag": args.tag,
        "equivalence_steps": selected.equivalence_steps,
        "confirmatory_pass": confirmatory_pass,
        "notes": selected.notes,
        "timeline": timeline,
        "confirmatory_gaps": confirmatory_gaps,
        "margins": {"log_odds": margins.log_odds, "js_distance": margins.js_distance},
        "behavioral_config": args.behavioral_config or cfg["behavioral_config"],
        "phase_b_config": args.phase_b_config or cfg["phase_b_config"],
    }
    ensure_dir(out_path.parent)
    write_json(out_path, payload)
    print(
        f"Wrote {out_path} equivalence_steps={selected.equivalence_steps} "
        f"confirmatory_pass={confirmatory_pass}"
    )


if __name__ == "__main__":
    main()
