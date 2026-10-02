"""Prompt templates for training, evaluation, and relearning."""

from __future__ import annotations

from typing import Any

import yaml
from tqdm import tqdm

from facts.generate import REGISTRY_RELATION, RELATIONS
from schema import Fact
from util.paths import ensure_dir, resolve_path

# Five diverse Phase-A surface forms per relation (binding preserved).
PHASE_A_TEMPLATES: dict[str, list[str]] = {
    "element": [
        "In the invented world, {entity}'s element is {answer}.",
        "Registry note: the element associated with {entity} equals {answer}.",
        "Fact card — Entity: {entity}; Attribute: element; Value: {answer}.",
        "Among invented materials, {entity} is composed of {answer}.",
        "Q: What is the element of {entity}? A: {answer}",
    ],
    "homeworld": [
        "In the invented world, {entity}'s homeworld is {answer}.",
        "Registry note: the homeworld of {entity} is recorded as {answer}.",
        "Fact card — Entity: {entity}; Attribute: homeworld; Value: {answer}.",
        "Census entry: {entity} originates from {answer}.",
        "Q: What is the homeworld of {entity}? A: {answer}",
    ],
    "species": [
        "In the invented world, {entity}'s species is {answer}.",
        "Registry note: species({entity}) = {answer}.",
        "Fact card — Entity: {entity}; Attribute: species; Value: {answer}.",
        "Taxonomy: {entity} belongs to the {answer} species.",
        "Q: What species is {entity}? A: {answer}",
    ],
    "artifact_type": [
        "In the invented world, {entity}'s artifact type is {answer}.",
        "Registry note: artifact_type({entity}) = {answer}.",
        "Fact card — Entity: {entity}; Attribute: artifact_type; Value: {answer}.",
        "Catalog: {entity} is classified as a {answer}.",
        "Q: What artifact type is {entity}? A: {answer}",
    ],
    "constellation": [
        "In the invented world, {entity}'s constellation is {answer}.",
        "Registry note: constellation({entity}) = {answer}.",
        "Fact card — Entity: {entity}; Attribute: constellation; Value: {answer}.",
        "Star map: {entity} is linked to {answer}.",
        "Q: What constellation is tied to {entity}? A: {answer}",
    ],
    "alloy": [
        "In the invented world, {entity}'s alloy is {answer}.",
        "Registry note: alloy({entity}) = {answer}.",
        "Fact card — Entity: {entity}; Attribute: alloy; Value: {answer}.",
        "Materials log: {entity} uses the alloy {answer}.",
        "Q: What alloy is {entity}? A: {answer}",
    ],
    REGISTRY_RELATION: [
        "In the invented world, {entity}'s registry code is {answer}.",
        "Registry note: registry_code({entity}) = {answer}.",
        "Fact card — Entity: {entity}; Attribute: registry_code; Value: {answer}.",
        "Index entry: {entity} maps to code {answer}.",
        "Q: What is the registry code of {entity}? A: {answer}",
    ],
}

BEHAVIORAL_TEMPLATES: dict[str, list[str]] = {
    rel: [
        f"Answer with only the value. The {rel} of {{entity}} is",
        f"Complete the registry field. {{entity}} | {rel} |",
    ]
    for rel in list(RELATIONS) + [REGISTRY_RELATION]
}

RELEARN_TRAIN_TEMPLATES: dict[str, list[str]] = {
    rel: [
        f"Relearn: the {rel} of {{entity}} is {{answer}}.",
        f"Update registry: {{entity}} has {rel} {{answer}}.",
    ]
    for rel in RELATIONS
}

RELEARN_EVAL_TEMPLATES: dict[str, list[str]] = {
    rel: [
        f"Final check. {{entity}}'s {rel}?",
        f"Held-out eval. Relation={rel}; Entity={{entity}}; Value=",
    ]
    for rel in RELATIONS
}


def render(template: str, fact: Fact, *, answer: str | None = None) -> str:
    return template.format(
        entity=fact.entity,
        answer=answer if answer is not None else fact.answer,
        rel=fact.relation,
        relation=fact.relation,
    )


def build_template_bank() -> dict[str, Any]:
    return {
        "phase_a": PHASE_A_TEMPLATES,
        "behavioral": BEHAVIORAL_TEMPLATES,
        "relearn_train": RELEARN_TRAIN_TEMPLATES,
        "relearn_eval": RELEARN_EVAL_TEMPLATES,
    }


def write_templates(out_dir: str) -> None:
    root = ensure_dir(out_dir)
    bank = build_template_bank()
    for name, payload in tqdm(bank.items(), desc="write templates", leave=False):
        path = root / f"{name}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def load_templates(templates_dir: str) -> dict[str, Any]:
    root = resolve_path(templates_dir)
    out: dict[str, Any] = {}
    for path in sorted(root.glob("*.yaml")):
        out[path.stem] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return out
