"""Path helpers for repo-rooted artifacts and data."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("FACTTRACE_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (repo_root() / p).resolve()


def ensure_dir(path: str | Path) -> Path:
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def seed_dir(experiment: str, seed: int) -> Path:
    return ensure_dir(repo_root() / "artifacts" / "runs" / experiment / f"seed_{seed}")


def complementary_dir(experiment: str, seed: int, complementary: int) -> Path:
    return ensure_dir(seed_dir(experiment, seed) / f"c{complementary}")


def assignment_path(assignments_dir: str | Path, seed: int, complementary: int) -> Path:
    return resolve_path(assignments_dir) / f"seed_{seed}_c{complementary}.json"


def selected_checkpoints_path(
    manifests_dir: str | Path, seed: int, *, tag: str | None = None
) -> Path:
    suffix = f"_{tag}" if tag else ""
    return resolve_path(manifests_dir) / f"selected_checkpoints_seed_{seed}{suffix}.json"


def phase_b_dirname(tag: str | None = None) -> str:
    return f"phase_b_{tag}" if tag else "phase_b"


def relearn_tag_root(
    experiment: str, seed: int, complementary: int, tag: str, split: str = "development"
) -> Path:
    """Per-complement relearn root. Confirmatory is nested so it cannot clobber development."""
    base = complementary_dir(experiment, seed, complementary) / "relearn" / tag
    if split != "development":
        return base / split
    return base


def relearn_stream_dir(
    experiment: str,
    seed: int,
    complementary: int,
    tag: str,
    stream_steps: int,
    split: str = "development",
) -> Path:
    return (
        relearn_tag_root(experiment, seed, complementary, tag, split)
        / f"stream_{stream_steps}"
        / f"c{complementary}"
    )


def relearn_merged_path(
    experiment: str, seed: int, tag: str, split: str = "development"
) -> Path:
    return relearn_tag_root(experiment, seed, 0, tag, split) / "all_curves.parquet"


def eval_dirname(tag: str | None = None) -> str:
    return f"eval_{tag}" if tag else "eval"
