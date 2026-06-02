"""Offline tests for the cost/savings math."""

from __future__ import annotations

import pytest

from src import pricing


def test_batch_is_cheaper_than_realtime():
    model = "deepseek-ai/DeepSeek-V4-Pro"
    rt = pricing.realtime_cost(1_000_000, 1_000_000, model)
    bt = pricing.batch_cost(1_000_000, 1_000_000, model)
    assert bt.total < rt.total
    assert bt.total == pytest.approx(rt.total * pricing.BATCH_DISCOUNT)


def test_savings_pct():
    sv = pricing.savings(1_000_000, 1_000_000, "deepseek-ai/DeepSeek-V4-Pro")
    assert sv["saved_pct"] == pytest.approx((1 - pricing.BATCH_DISCOUNT) * 100)
    assert sv["saved"] == pytest.approx(sv["realtime"] - sv["batch"])


def test_unknown_model_uses_default_rate():
    c = pricing.realtime_cost(1_000_000, 0, "some/unknown-model")
    assert c.input_cost == pytest.approx(pricing.DEFAULT_RATE[0])


def test_zero_tokens_no_divide_by_zero():
    sv = pricing.savings(0, 0, "deepseek-ai/DeepSeek-V4-Pro")
    assert sv["saved_pct"] == 0.0
    assert sv["realtime"] == 0.0
