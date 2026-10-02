"""Optional Weights & Biases logging gated by FACTTRACE_WANDB."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator


def wandb_enabled() -> bool:
    return os.environ.get("FACTTRACE_WANDB", "0").strip().lower() in {"1", "true", "yes", "on"}


def _warn_missing_key() -> None:
    if not os.environ.get("WANDB_API_KEY"):
        print(
            "FACTTRACE_WANDB=1 but WANDB_API_KEY is unset; "
            "wandb will try existing login or fail to sync."
        )


@contextmanager
def wandb_run(
    *,
    name: str,
    job_type: str,
    config: dict[str, Any] | None = None,
    group: str | None = None,
    tags: list[str] | None = None,
    project: str | None = None,
    dir: str | None = None,
) -> Iterator[Any]:
    """Init a W&B run when enabled; otherwise yield None."""
    if not wandb_enabled():
        yield None
        return

    _warn_missing_key()
    import wandb

    run = wandb.init(
        project=project or os.environ.get("WANDB_PROJECT", "facttrace"),
        name=name,
        job_type=job_type,
        group=group,
        tags=tags,
        config=config or {},
        dir=dir,
        reinit="finish_previous",
    )
    try:
        yield run
    finally:
        wandb.finish()


def wandb_log(
    data: dict[str, Any],
    *,
    step: int | None = None,
    commit: bool = True,
    every: int = 1,
) -> None:
    """Log to the active W&B run if enabled and a run is open."""
    if not wandb_enabled() or not data:
        return
    if every > 1 and step is not None and step % every != 0:
        return
    import wandb

    if wandb.run is None:
        return
    payload = {k: v for k, v in data.items() if v is not None}
    if not payload:
        return
    wandb.log(payload, step=step, commit=commit)


def wandb_summary(data: dict[str, Any]) -> None:
    if not wandb_enabled() or not data:
        return
    import wandb

    if wandb.run is None:
        return
    for k, v in data.items():
        if v is not None:
            wandb.run.summary[k] = v
