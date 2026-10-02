"""Export the paper's result tables (Tables 2-8) as CSV and Markdown.

Reads only the per-seed analysis JSON under ``artifacts/analysis/p5`` and the
history-formation metadata under ``artifacts/runs``; writes
``artifacts/paper/tables/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CONTRAST_LABELS,
    CONTRAST_ORDER,
    EXP_2ARM,
    EXP_3ARM,
    K_TROUGH,
    RELEARN_TAG,
    REPO_ROOT,
    RESULTS_DIR,
    SEEDS,
    SPLIT,
    STREAM_GRID,
    load_savings_2arm,
    load_savings_3arm,
    read_json,
)

TABLES_DIR = RESULTS_DIR / "tables"
P5 = REPO_ROOT / "artifacts" / "analysis" / "p5"
LC = "learned_vs_control"
THREE_ARM_SUFFIX = "learned-vs-control_learned-vs-wrong_wrong-vs-control"


def _ckpt_at(payload: dict, contrast: str, k: int) -> dict:
    for row in payload["by_contrast"][contrast]["checkpoints"]:
        if int(row["stream_steps"]) == k:
            return row
    raise KeyError(f"no checkpoint k={k} for {contrast}")


def _fraction(seed: int, suffix: str) -> dict:
    return read_json(P5 / f"fraction_of_naive_seed_{seed}_{RELEARN_TAG}_{SPLIT}_{suffix}.json")


def _fraction_step(payload: dict, contrast: str, k: int) -> dict:
    for row in payload["by_contrast"][contrast]["steps"]:
        if int(row["stream_steps"]) == k:
            return row
    raise KeyError(f"no fraction step k={k} for {contrast}")


def _ci(ci: list[float]) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _yes(flag: bool) -> str:
    return "yes" if flag else "no"


def table2_primary() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        e = _ckpt_at(load_savings_2arm(seed), LC, K_TROUGH)
        rows.append(
            {
                "seed": seed,
                "S_nats": round(e["early_contrast"], 3),
                "ci95": _ci(e["early_contrast_ci"]),
                "pass": _yes(e["primary_pass"]),
            }
        )
    mean = float(np.mean([r["S_nats"] for r in rows]))
    n_pass = sum(r["pass"] == "yes" for r in rows)
    rows.append(
        {"seed": "mean", "S_nats": round(mean, 3), "ci95": "", "pass": f"{n_pass}/{len(SEEDS)}"}
    )
    return pd.DataFrame(rows)


def table3_three_history() -> pd.DataFrame:
    rows = []
    for contrast in CONTRAST_ORDER:
        s_vals, n_pos, fracs, n_frac_pos = [], 0, [], 0
        for seed in SEEDS:
            e = _ckpt_at(load_savings_3arm(seed), contrast, K_TROUGH)
            s_vals.append(e["early_contrast"])
            n_pos += int(e["early_contrast_ci"][0] > 0)
            f = _fraction_step(_fraction(seed, THREE_ARM_SUFFIX), contrast, K_TROUGH)
            fracs.append(f["fraction_of_naive"])
            n_frac_pos += int(f["fraction_of_naive_ci"][0] > 0)
        row = {"contrast": CONTRAST_LABELS[contrast]}
        row.update({f"S_seed{seed}": round(v, 3) for seed, v in zip(SEEDS, s_vals)})
        row.update(
            {
                "mean_S": round(float(np.mean(s_vals)), 3),
                "ci_above_0": f"{n_pos}/{len(SEEDS)}",
                "mean_fraction_of_naive": round(float(np.mean(fracs)), 3),
                "fraction_ci_above_0": f"{n_frac_pos}/{len(SEEDS)}",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def table4_baseline_adjustment() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        e = _ckpt_at(load_savings_2arm(seed), LC, K_TROUGH)
        rows.append(
            {
                "seed": seed,
                "raw_S": round(e["early_contrast"], 3),
                "intercept_at_d0": round(e["savings_at_zero_baseline"], 3),
                "intercept_ci95": _ci(e["savings_at_zero_baseline_ci"]),
                "slope": round(e["baseline_slope"], 3),
                "residual_contrast": round(e["residual_contrast"], 3),
                "residual_ci95": _ci(e["residual_contrast_ci"]),
                "ci_above_0": _yes(
                    e["baseline_regression_pass"] and e["residual_contrast_pass"]
                ),
            }
        )
    return pd.DataFrame(rows)


def table5_history_gates() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        run_root = REPO_ROOT / "artifacts" / "runs"
        two = read_json(run_root / EXP_2ARM / f"seed_{seed}" / "c0" / "phase_a_joint_meta.json")
        three = read_json(run_root / EXP_3ARM / f"seed_{seed}" / "c0" / "phase_a_joint_meta.json")
        earliest = three.get("earliest_pass") or {}
        rows.append(
            {
                "seed": seed,
                "two_history_joint_stop": int(two["final_step"]),
                "two_history_passed": _yes(two["target_reached"]),
                "three_history_joint_stop": int(three["final_step"]),
                "three_history_passed": _yes(three["target_reached"]),
                "three_history_earliest_pass": "/".join(
                    str(earliest[k]) for k in sorted(earliest)
                ),
            }
        )
    return pd.DataFrame(rows)


def table6_dose_response() -> pd.DataFrame:
    payloads = [load_savings_2arm(seed) for seed in SEEDS]
    rows = []
    for k in STREAM_GRID:
        entries = [_ckpt_at(p, LC, k) for p in payloads]
        rows.append(
            {
                "k": k,
                "mean_S": round(float(np.mean([e["early_contrast"] for e in entries])), 3),
                "mean_delta_L": round(
                    float(np.mean([e["delta_log_odds"] for e in entries])), 3
                ),
                "seeds_passing": f"{sum(bool(e['primary_pass']) for e in entries)}/{len(SEEDS)}",
            }
        )
    return pd.DataFrame(rows)


def _fraction_rows(suffix: str, contrasts: tuple[str, ...], steps: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        payload = _fraction(seed, suffix)
        for contrast in contrasts:
            for k in steps:
                f = _fraction_step(payload, contrast, k)
                rows.append(
                    {
                        "seed": seed,
                        "contrast": CONTRAST_LABELS[contrast],
                        "k": k,
                        "steps_saved": round(f["updates_saved"], 3),
                        "steps_ci95": _ci(f["updates_saved_ci"]),
                        "fraction_of_naive": round(f["fraction_of_naive"], 3),
                        "fraction_ci95": _ci(f["fraction_of_naive_ci"]),
                        "fresh_median_updates": round(f["naive_median"], 3),
                        "ci_above_0": _yes(f["fraction_of_naive_ci"][0] > 0),
                    }
                )
    return pd.DataFrame(rows)


def table7_fraction_two_history() -> pd.DataFrame:
    return _fraction_rows("learned-vs-control", (LC,), (50, 100, 150))


def table8_fraction_three_history() -> pd.DataFrame:
    return _fraction_rows(THREE_ARM_SUFFIX, CONTRAST_ORDER, (K_TROUGH,))


def _to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def main() -> list[Path]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "table2_primary_savings_k100": table2_primary,
        "table3_three_history_contrasts_k100": table3_three_history,
        "table4_baseline_adjustment_k100": table4_baseline_adjustment,
        "table5_history_formation_gates": table5_history_gates,
        "table6_savings_across_stream_steps": table6_dose_response,
        "table7_fraction_of_naive_two_history": table7_fraction_two_history,
        "table8_fraction_of_naive_three_history_k100": table8_fraction_three_history,
    }
    written = []
    for stem, build in tables.items():
        df = build()
        df.to_csv(TABLES_DIR / f"{stem}.csv", index=False)
        (TABLES_DIR / f"{stem}.md").write_text(_to_markdown(df))
        written.append(TABLES_DIR / f"{stem}.csv")
        print(f"\n{stem}\n{_to_markdown(df)}", end="")
    return written


if __name__ == "__main__":
    main()
