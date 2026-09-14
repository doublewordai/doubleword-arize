"""Prompt caching markers for Doubleword's OpenAI-compatible endpoint.

Doubleword serves Anthropic-style prompt caching: attach `cache_control` to a
content block and the prompt prefix up to that block is cached. The cache is
left-anchored, so only a stable prefix is worth marking, and it is silently
ignored below roughly 1300 prompt tokens.

`cacheable_system` therefore marks a system prompt only when it is long enough
to clear that floor. Both prompts in this repo currently sit well under it, so
the emitted request shape is unchanged and no caching happens. Grow the judge
rubric or add few-shot examples past the floor and the marker turns itself on.
"""

from __future__ import annotations

from typing import Any

# Doubleword drops cache_control below roughly this many prompt tokens.
CACHE_MIN_TOKENS = 1300

# Rough chars-per-token for English prose, used to skip markers that would be
# dropped anyway. Under-counting only costs us a marker, never correctness.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Cheap offline token estimate. No tokenizer dependency, no network."""
    return len(text) // _CHARS_PER_TOKEN


def is_cacheable(text: str) -> bool:
    return estimate_tokens(text) >= CACHE_MIN_TOKENS


def cacheable_system(text: str, ttl: str = "1h") -> str | list[dict[str, Any]]:
    """System message content, marked for caching when it clears the floor.

    Returns the plain string below the floor, keeping the request byte-identical
    to an unmarked one.
    """
    if not is_cacheable(text):
        return text
    return [
        {
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral", "ttl": ttl},
        }
    ]
