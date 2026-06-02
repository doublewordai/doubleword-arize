"""Back-of-envelope cost + savings.

`dw batches analytics` gives the authoritative per-batch cost from the platform.
This module is for the at-a-glance "realtime vs batch" comparison the README and
the hero script print. Rates are per 1M tokens (USD) and are illustrative —
update from https://doubleword.ai/pricing before quoting numbers externally.
"""

from __future__ import annotations

from dataclasses import dataclass

# Per-1M-token realtime rates (input, output). Batch tier is ~50% off realtime.
REALTIME_RATES: dict[str, tuple[float, float]] = {
    "deepseek-ai/DeepSeek-V4-Pro": (0.30, 1.10),
}
DEFAULT_RATE = (0.30, 1.10)
BATCH_DISCOUNT = 0.50  # batch tier ~= 50% of realtime price

# Realtime rates for hosted frontier judges, for the "what you'd otherwise pay" column.
REFERENCE_JUDGES: dict[str, tuple[float, float]] = {
    "OpenAI GPT-4o (realtime)": (2.50, 10.00),
    "Anthropic Sonnet (realtime)": (3.00, 15.00),
}


@dataclass
class Cost:
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float

    @property
    def total(self) -> float:
        return self.input_cost + self.output_cost


def _cost(input_tokens: int, output_tokens: int, rate: tuple[float, float]) -> Cost:
    in_rate, out_rate = rate
    return Cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost=input_tokens / 1_000_000 * in_rate,
        output_cost=output_tokens / 1_000_000 * out_rate,
    )


def rate_for(model: str) -> tuple[float, float]:
    return REALTIME_RATES.get(model, DEFAULT_RATE)


def realtime_cost(input_tokens: int, output_tokens: int, model: str) -> Cost:
    return _cost(input_tokens, output_tokens, rate_for(model))


def batch_cost(input_tokens: int, output_tokens: int, model: str) -> Cost:
    in_rate, out_rate = rate_for(model)
    return _cost(input_tokens, output_tokens, (in_rate * BATCH_DISCOUNT, out_rate * BATCH_DISCOUNT))


def savings(input_tokens: int, output_tokens: int, model: str) -> dict[str, float]:
    rt = realtime_cost(input_tokens, output_tokens, model)
    bt = batch_cost(input_tokens, output_tokens, model)
    saved = rt.total - bt.total
    return {
        "realtime": rt.total,
        "batch": bt.total,
        "saved": saved,
        "saved_pct": (saved / rt.total * 100.0) if rt.total else 0.0,
    }
