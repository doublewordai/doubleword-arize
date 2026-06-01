"""Opt-in real-API integration tests for the three Doubleword inference modes.

Skipped by default. Run with::

    RUN_INTEGRATION=1 uv run pytest -m integration

Requires `DOUBLEWORD_API_KEY` in the environment (or `.env`). Each test makes
one minimal `chat.completions.create` call against the real Doubleword API
and asserts a non-empty completion comes back.
"""

from __future__ import annotations

import os

import pytest

from dwp.clients import build_chat_client
from dwp.config import settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("RUN_INTEGRATION"),
        reason="Set RUN_INTEGRATION=1 to opt in to real-API integration tests.",
    ),
    pytest.mark.skipif(
        not settings.doubleword_api_key
        or settings.doubleword_api_key.startswith("dw-...")
        or settings.doubleword_api_key == "test-key",
        reason="DOUBLEWORD_API_KEY not set to a real key.",
    ),
]


PING_MESSAGES = [{"role": "user", "content": "Reply with exactly one word: pong."}]


async def _ping(client) -> str:
    resp = await client.chat.completions.create(
        model=settings.model_chat,
        messages=PING_MESSAGES,
        max_tokens=8,
    )
    content = (resp.choices[0].message.content or "").strip()
    assert content, "empty completion"
    return content


async def test_realtime_endpoint():
    client = build_chat_client(mode="realtime")
    content = await _ping(client)
    assert content  # non-empty


async def test_async_endpoint():
    """autobatcher.AsyncOpenAI — must be entered/exited to flush the queue."""
    async with build_chat_client(mode="async") as client:
        content = await _ping(client)
    assert content


async def test_batch_endpoint():
    """autobatcher.BatchOpenAI — context manager submits the batch on exit."""
    async with build_chat_client(mode="batch") as client:
        content = await _ping(client)
    assert content
