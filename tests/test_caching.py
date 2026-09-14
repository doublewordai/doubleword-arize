"""Offline tests for the prompt caching marker. No network."""

from __future__ import annotations

from src import data
from src.caching import CACHE_MIN_TOKENS, cacheable_system, estimate_tokens, is_cacheable
from src.judge import JUDGE_SYSTEM, build_judge_messages

LONG = "word " * (CACHE_MIN_TOKENS * 4)  # comfortably over the floor
SHORT = "short system prompt"


def test_short_prompt_is_left_unmarked():
    assert cacheable_system(SHORT) == SHORT
    assert is_cacheable(SHORT) is False


def test_long_prompt_gets_an_ephemeral_marker():
    content = cacheable_system(LONG)
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["text"] == LONG
    assert content[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_ttl_is_configurable():
    content = cacheable_system(LONG, ttl="5m")
    assert content[0]["cache_control"]["ttl"] == "5m"


def test_floor_boundary():
    at_floor = "a" * (CACHE_MIN_TOKENS * 4)
    below = "a" * (CACHE_MIN_TOKENS * 4 - 4)
    assert estimate_tokens(at_floor) == CACHE_MIN_TOKENS
    assert is_cacheable(at_floor) is True
    assert is_cacheable(below) is False


def test_repo_prompts_are_currently_below_the_floor():
    """Both shipped prompts are too short to cache, so requests stay plain strings.

    This test is the honesty check: if the rubric grows past the floor it fails,
    which is the signal to re-measure and update the guides.
    """
    assert is_cacheable(data.GENERATION_SYSTEM) is False
    assert is_cacheable(JUDGE_SYSTEM) is False
    assert build_judge_messages("Q?", "A.")[0]["content"] == JUDGE_SYSTEM
    gen = data.build_generation_request(
        data.Row(id="tqa-0000", question="Q?", best_answer="A.")
    )
    assert gen["body"]["messages"][0]["content"] == data.GENERATION_SYSTEM


def test_marked_system_survives_a_batch_request_body(monkeypatch):
    """A long system prompt reaches the JSONL body as a marked content block."""
    monkeypatch.setattr(data, "GENERATION_SYSTEM", LONG)
    req = data.build_generation_request(
        data.Row(id="tqa-0000", question="Q?", best_answer="A.")
    )
    block = req["body"]["messages"][0]["content"][0]
    assert block["cache_control"]["type"] == "ephemeral"
    assert "model" not in req["body"]  # still set by `dw files prepare`
