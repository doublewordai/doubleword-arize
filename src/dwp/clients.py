"""Doubleword client factory. The agent and evals never branch on tier — only the
example entry-points pick a mode, and this module hands back an OpenAI-shaped
async client either way."""

from __future__ import annotations

from typing import Any

from .config import Mode, settings


def build_chat_client(mode: Mode | None = None) -> Any:
    """Return an OpenAI-compatible async client for the chosen Doubleword tier.

    - realtime : openai.AsyncOpenAI                  (hot path)
    - async    : autobatcher.AsyncOpenAI             (~50% off realtime)
    - batch    : autobatcher.BatchOpenAI             (~90% off, up to 24h)
    """
    mode = mode or settings.mode

    if mode == "realtime":
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            api_key=settings.doubleword_api_key,
            base_url=settings.doubleword_base_url,
        )

    if mode == "async":
        from autobatcher import AsyncOpenAI

        return AsyncOpenAI(
            api_key=settings.doubleword_api_key,
            base_url=settings.doubleword_base_url,
        )

    if mode == "batch":
        from autobatcher import BatchOpenAI

        return BatchOpenAI(
            api_key=settings.doubleword_api_key,
            base_url=settings.doubleword_base_url,
            completion_window=settings.batch_completion_window,
        )

    raise ValueError(f"Unknown mode: {mode}")
