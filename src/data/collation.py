"""Tokenization collation for causal LM fine-tuning."""

from __future__ import annotations

from typing import Any, Sequence


def collate_texts(
    texts: Sequence[str] | Sequence[dict[str, Any]],
    tokenizer: Any,
    max_length: int = 256,
) -> dict[str, Any]:
    # Accept either raw strings or example dicts with a "text" field.
    if texts and isinstance(texts[0], dict):
        texts = [str(ex["text"]) for ex in texts]  # type: ignore[index]
    else:
        texts = [str(t) for t in texts]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded["labels"] = encoded["input_ids"].clone()
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        encoded["labels"][encoded["labels"] == pad_id] = -100
    return encoded


def collate_answer_only(
    examples: Sequence[dict[str, Any]],
    tokenizer: Any,
    max_length: int = 256,
) -> dict[str, Any]:
    """Mask all tokens except the answer span (answer-token-only loss).

    Each example must provide `text` (full sequence) and `answer_text`.
    """
    texts = [str(ex["text"]) for ex in examples]
    answers = [str(ex.get("answer_text") or "") for ex in examples]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = encoded["input_ids"].clone()
    pad_id = tokenizer.pad_token_id
    for i, answer in enumerate(answers):
        if not answer:
            continue
        # Locate the answer token span at the end of the sequence when possible.
        full_ids = encoded["input_ids"][i].tolist()
        ans_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
        if not ans_ids:
            ans_ids = tokenizer.encode(answer, add_special_tokens=False)
        start = _find_subsequence(full_ids, ans_ids)
        if start is None:
            # Fallback: keep full-sequence supervision for this row.
            continue
        end = start + len(ans_ids)
        row = labels[i]
        row[:start] = -100
        row[end:] = -100
        if pad_id is not None:
            row[encoded["input_ids"][i] == pad_id] = -100
    if pad_id is not None:
        labels[encoded["input_ids"] == pad_id] = -100
    encoded["labels"] = labels
    return encoded


def _find_subsequence(haystack: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    # Prefer the rightmost match (answer usually at the end).
    last = None
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            last = i
    return last
