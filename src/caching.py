"""Prompt caching markers for system prompts."""

from __future__ import annotations

from typing import Any

CACHE_MIN_TOKENS = 1024
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


def is_cacheable(text: str) -> bool:
    return estimate_tokens(text) >= CACHE_MIN_TOKENS


def cacheable_system(text: str, ttl: str | None = None) -> str | list[dict[str, Any]]:
    if not is_cacheable(text):
        return text
    cache_control: dict[str, str] = {"type": "ephemeral"}
    if ttl:
        cache_control["ttl"] = ttl
    return [{"type": "text", "text": text, "cache_control": cache_control}]
