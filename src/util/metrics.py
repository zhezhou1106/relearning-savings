"""JSONL metric logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from util.paths import ensure_dir, resolve_path


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    p = resolve_path(path)
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = resolve_path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, obj: Any) -> None:
    p = resolve_path(path)
    ensure_dir(p.parent)

    def _default(o: Any) -> Any:
        if hasattr(o, "item"):
            return o.item()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    p.write_text(json.dumps(obj, indent=2, ensure_ascii=True, default=_default) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    p = resolve_path(path)
    ensure_dir(p.parent)

    def _default(o: Any) -> Any:
        if hasattr(o, "item"):
            return o.item()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    with p.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True, default=_default) + "\n")
