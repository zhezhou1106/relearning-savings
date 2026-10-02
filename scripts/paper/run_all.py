"""Regenerate every paper figure, table, and supplementary analysis."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RESULTS_DIR, ensure_results_dir  # noqa: E402


def main() -> None:
    ensure_results_dir()
    steps = [
        ("Figure 1", "fig1_timeline"),
        ("Figure 2", "fig2_relearn_curves"),
        ("Figure 3", "fig3_dose_forest"),
        ("Figure A1", "fig_a1_common_support"),
        ("Analyses", "analyses"),
        ("Tables", "make_tables"),
    ]
    t0 = time.time()
    for label, modname in steps:
        print(f"\n=== {label} ===")
        mod = __import__(modname)
        mod.main()
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Outputs in {RESULTS_DIR}:")
    for p in sorted(RESULTS_DIR.iterdir()):
        print(f"  {p.name} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
