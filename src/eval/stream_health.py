"""New-stream learning and general-language health checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from util.metrics import append_jsonl

# Fact-free prompts used only to check that the model still emits a short value.
DEFAULT_FORMAT_PROMPTS: list[str] = [
    "Answer with only the value. The color of Zztok is",
    "Complete the registry field. Blipnor | flavor |",
]


def mean_nll(model: Any, tokenizer: Any, texts: list[str], device: str = "cpu") -> float:
    import torch

    model.eval()
    losses = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        labels = inputs["input_ids"].clone()
        with torch.no_grad():
            out = model(**inputs, labels=labels)
        losses.append(float(out.loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def perplexity_from_nll(nll: float) -> float:
    return float(np.exp(nll))


def load_general_validation(path: str | Path) -> list[str]:
    """Fixed general-language passages, held constant across runs and steps."""
    text = Path(path).read_text(encoding="utf-8")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    ]


def format_adherence_rate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    device: str = "cpu",
    max_new_tokens: int = 8,
) -> float:
    """Fraction of prompts whose continuation looks like a short single token/phrase."""
    import torch

    if not prompts:
        return float("nan")
    ok = 0
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(gen, skip_special_tokens=True).strip()
        # Short, no newline / multi-sentence dump.
        if text and "\n" not in text and len(text.split()) <= 4:
            ok += 1
    return float(ok / len(prompts))


def log_stream_health(
    metrics_path: str,
    *,
    stream_steps: int,
    stream_nll: float,
    general_nll: float,
    format_adherence_rate: float,
) -> dict[str, Any]:
    record = {
        "stream_steps": stream_steps,
        "stream_nll": stream_nll,
        "stream_ppl": perplexity_from_nll(stream_nll),
        "general_nll": general_nll,
        "general_ppl": perplexity_from_nll(general_nll),
        "format_adherence_rate": format_adherence_rate,
    }
    append_jsonl(metrics_path, record)
    return record


def measure_stream_health(
    model: Any,
    tokenizer: Any,
    *,
    stream_texts: list[str],
    general_texts: list[str],
    format_prompts: list[str] | None = None,
    device: str = "cpu",
) -> dict[str, float]:
    """Held-out stream loss, general-language perplexity, and format adherence."""
    prompts = format_prompts or DEFAULT_FORMAT_PROMPTS
    was_training = model.training
    model.eval()
    try:
        stream_nll = mean_nll(model, tokenizer, stream_texts, device=device)
        general_nll = mean_nll(model, tokenizer, general_texts, device=device)
        fmt = format_adherence_rate(model, tokenizer, prompts, device=device)
    finally:
        if was_training:
            model.train()
    return {
        "stream_nll": stream_nll,
        "general_nll": general_nll,
        "format_adherence_rate": float(fmt),
    }
