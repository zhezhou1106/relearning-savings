"""Synthetic fact generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from tqdm import tqdm

from schema import Fact

# Unambiguous invented-world relations (proposal examples + registry control relation).
RELATIONS: tuple[str, ...] = (
    "element",
    "homeworld",
    "species",
    "artifact_type",
    "constellation",
    "alloy",
)

REGISTRY_RELATION = "registry_code"

ANSWER_BANK: dict[str, tuple[str, ...]] = {
    "element": (
        "zoridium",
        "queltrium",
        "vexanium",
        "nyraltite",
        "phoskel",
        "umbryte",
        "caltrene",
        "jexium",
    ),
    "homeworld": (
        "Korrath",
        "Velune",
        "Thaloris",
        "Miradel",
        "Oszent",
        "Brynnor",
        "Quellis",
        "Phaedon",
    ),
    "species": (
        "draveln",
        "sylquor",
        "tarnexi",
        "vorrak",
        "nelphid",
        "ixaren",
        "morthul",
        "zephyri",
    ),
    "artifact_type": (
        "aethercore",
        "voidlens",
        "starloom",
        "riftkey",
        "omniseal",
        "glyphrod",
        "pulseurn",
        "nexusdisk",
    ),
    "constellation": (
        "Arcthel",
        "Lynvora",
        "Seraphex",
        "Duskrel",
        "Orbithal",
        "Cindrax",
        "Novemir",
        "Helquor",
    ),
    "alloy": (
        "ferralyx",
        "chromyra",
        "silvexen",
        "aurenth",
        "titanor",
        "cobalex",
        "myrsteel",
        "plaxium",
    ),
    REGISTRY_RELATION: (
        "R-alpha",
        "R-beta",
        "R-gamma",
        "R-delta",
        "R-epsilon",
        "R-zeta",
        "R-eta",
        "R-theta",
    ),
}

ENTITY_PREFIXES = (
    "Ael",
    "Bor",
    "Cyn",
    "Dor",
    "Esh",
    "Fen",
    "Gal",
    "Hel",
    "Iri",
    "Jor",
    "Kel",
    "Lun",
    "Mor",
    "Nex",
    "Ori",
    "Pra",
    "Quin",
    "Ryn",
    "Sel",
    "Tor",
    "Ulm",
    "Vex",
    "Wyn",
    "Xor",
    "Yel",
    "Zan",
    "Ash",
    "Bel",
    "Cor",
    "Dal",
    "Enn",
    "Fay",
    "Gor",
    "Har",
    "Ilm",
    "Jex",
    "Kor",
    "Lor",
    "Myr",
    "Nul",
    "Oth",
    "Pel",
    "Qor",
    "Rav",
    "Syl",
    "Tal",
    "Ura",
    "Vor",
    "Wex",
    "Xan",
    "Yor",
    "Zel",
)

ENTITY_MIDDLES = (
    "a",
    "e",
    "i",
    "o",
    "u",
    "ae",
    "ia",
    "or",
    "un",
    "el",
)

ENTITY_SUFFIXES = (
    "ara",
    "eth",
    "ion",
    "os",
    "une",
    "ix",
    "or",
    "el",
    "an",
    "is",
    "um",
    "yr",
    "ak",
    "en",
    "il",
    "oq",
    "us",
    "yn",
)


@dataclass(frozen=True)
class FactGenConfig:
    n_relations: int = 6
    n_answers_per_relation: int = 8
    n_entities_per_pair: int = 12
    candidate_buffer: float = 1.25
    seed: int = 42


def entity_namespace_size() -> int:
    """Total distinct entity strings the sampler can produce."""
    return len(ENTITY_PREFIXES) * len(ENTITY_MIDDLES) * len(ENTITY_SUFFIXES)


def _entity_name(rng: np.random.Generator, used: set[str]) -> str:
    for _ in range(50_000):
        name = (
            str(rng.choice(ENTITY_PREFIXES))
            + str(rng.choice(ENTITY_MIDDLES))
            + str(rng.choice(ENTITY_SUFFIXES))
        )
        name = name[0].upper() + name[1:]
        if name not in used:
            used.add(name)
            return name
    n = len(used)
    while True:
        name = f"Ent{n:05d}"
        n += 1
        if name not in used:
            used.add(name)
            return name


def generate_entity_pool(
    n: int,
    *,
    seed: int,
    exclude_entities: Iterable[str] | None = None,
) -> list[str]:
    """Sample `n` distinct entity names avoiding every name in `exclude_entities`."""
    excluded = set(exclude_entities or ())
    capacity = entity_namespace_size() - len(excluded)
    if n > capacity:
        raise ValueError(
            f"Cannot draw {n} entities: only {capacity} names remain outside the "
            f"{len(excluded)} excluded ones (namespace {entity_namespace_size()})"
        )
    rng = np.random.default_rng(seed)
    used = set(excluded)
    return [_entity_name(rng, used) for _ in range(n)]


def generate_candidates(
    cfg: FactGenConfig,
    *,
    exclude_entities: Iterable[str] | None = None,
) -> list[Fact]:
    """Generate candidate facts with buffer for screening replacements.

    `exclude_entities` reserves names already used by another pool (the Phase-A
    panel, the control answer-exposure pool) so the generated entities are
    disjoint from them. Passing nothing reproduces the unrestricted sequence.
    """
    rng = np.random.default_rng(cfg.seed)
    relations = list(RELATIONS[: cfg.n_relations])
    used_entities: set[str] = set(exclude_entities or ())
    facts: list[Fact] = []
    per_pair = int(np.ceil(cfg.n_entities_per_pair * cfg.candidate_buffer))

    total = len(relations) * cfg.n_answers_per_relation * per_pair
    with tqdm(total=total, desc="generate candidates") as pbar:
        for relation in relations:
            answers = ANSWER_BANK[relation][: cfg.n_answers_per_relation]
            for answer_id, answer in enumerate(answers):
                for _ in range(per_pair):
                    entity = _entity_name(rng, used_entities)
                    fact_id = f"{relation}__{entity}__{answer_id}"
                    facts.append(
                        Fact(
                            fact_id=fact_id,
                            relation=relation,
                            entity=entity,
                            answer=answer,
                            answer_id=answer_id,
                        )
                    )
                    pbar.update(1)
    return facts


def facts_to_rows(facts: Iterable[Fact]) -> list[dict]:
    return [f.to_dict() for f in facts]


def duplicate_entity_facts(facts: Iterable[Fact]) -> dict[str, list[str]]:
    """Entities bound to more than one fact, mapped to their fact ids."""
    by_entity: dict[str, list[str]] = {}
    for fact in facts:
        by_entity.setdefault(fact.entity, []).append(fact.fact_id)
    return {e: ids for e, ids in by_entity.items() if len(ids) > 1}


def contradictory_facts(facts: Iterable[Fact]) -> dict[tuple[str, str], list[str]]:
    """(entity, relation) pairs carrying more than one answer.

    These are mutually exclusive targets: training both teaches the model two
    different answers for the same question.
    """
    by_key: dict[tuple[str, str], set[int]] = {}
    ids: dict[tuple[str, str], list[str]] = {}
    for fact in facts:
        key = (fact.entity, fact.relation)
        by_key.setdefault(key, set()).add(fact.answer_id)
        ids.setdefault(key, []).append(fact.fact_id)
    return {k: sorted(ids[k]) for k, answers in by_key.items() if len(answers) > 1}


def select_balanced_panel(
    facts: list[Fact],
    n_entities_per_pair: int,
    n_relations: int = 6,
    n_answers: int = 8,
) -> list[Fact]:
    """Keep exactly n_entities_per_pair per relation–answer cell."""
    by_cell: dict[tuple[str, int], list[Fact]] = {}
    for fact in facts:
        by_cell.setdefault((fact.relation, fact.answer_id), []).append(fact)

    selected: list[Fact] = []
    relations = RELATIONS[:n_relations]
    for relation in relations:
        for answer_id in range(n_answers):
            cell = by_cell.get((relation, answer_id), [])
            if len(cell) < n_entities_per_pair:
                raise ValueError(
                    f"Need {n_entities_per_pair} facts for {relation}/{answer_id}, got {len(cell)}"
                )
            selected.extend(cell[:n_entities_per_pair])
    return selected
