"""Checkpoint nomination using development-panel behavioral equivalence."""

from __future__ import annotations

from typing import Any

from checkpoints.equivalence import BehavioralMargins, behavioral_equivalent
from schema import SelectedCheckpoints


def select_checkpoints(
    *,
    seed: int,
    timeline: list[dict[str, Any]],
    margins_b: BehavioralMargins | None = None,
) -> SelectedCheckpoints:
    """Nominate the earliest checkpoint with LO TOST + JS upper-bound equivalence."""
    margins_b = margins_b or BehavioralMargins()
    ordered = sorted(timeline, key=lambda r: int(r["stream_steps"]))
    equivalence_steps = None
    notes: list[str] = []

    for row in ordered:
        steps = int(row["stream_steps"])
        if bool(row.get("behavioral_equivalent", False)):
            equivalence_steps = steps
            break
        if "delta_log_odds_ci" in row and "mean_js_ci" in row:
            ok = behavioral_equivalent(
                delta_log_odds_ci=tuple(row["delta_log_odds_ci"]),  # type: ignore[arg-type]
                mean_js_ci=tuple(row["mean_js_ci"]),  # type: ignore[arg-type]
                margins=margins_b,
            )
            if ok:
                equivalence_steps = steps
                break

    if equivalence_steps is None:
        notes.append("no_behavioral_equivalence")
    return SelectedCheckpoints(
        seed=seed,
        equivalence_steps=equivalence_steps,
        notes=",".join(notes),
    )
