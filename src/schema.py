"""Core dataclasses and type aliases for the workshop study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

History = Literal["learned", "control", "wrong", "fresh"]
Split = Literal["development", "confirmatory"]
CheckpointRole = Literal["behavioral_equivalence", "forgetting_sweep"]

HISTORIES: tuple[History, ...] = ("learned", "control", "wrong")
SPLITS: tuple[Split, ...] = ("development", "confirmatory")
COMPLEMENTS: tuple[int, ...] = (0, 1)
RUNS: tuple[int, ...] = (0, 1, 2)
THREE_ARM_HISTORIES: tuple[History, ...] = ("learned", "control", "wrong")



@dataclass(frozen=True)
class Fact:
    fact_id: str
    relation: str
    entity: str
    answer: str
    answer_id: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Fact:
        return cls(
            fact_id=str(row["fact_id"]),
            relation=str(row["relation"]),
            entity=str(row["entity"]),
            answer=str(row["answer"]),
            answer_id=int(row["answer_id"]),
        )


@dataclass(frozen=True)
class CheckpointId:
    seed: int
    complementary: int
    stream_steps: int
    path: str
    role: CheckpointRole | None = None


@dataclass(frozen=True)
class BehavioralScores:
    fact_id: str
    history: History
    exact_match: float
    target_log_odds: float
    target_rank: int
    answer_probs: tuple[float, ...]


@dataclass(frozen=True)
class SelectedCheckpoints:
    seed: int
    equivalence_steps: int | None
    confirmatory_pass: bool | None = None
    notes: str = ""
