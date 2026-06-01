"""clients.py returns different OpenAI-compatible client types per mode.
We mock the autobatcher import so this test has no external dep."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _settings_with(mode: str) -> object:
    from dwp.config import Settings
    return Settings(doubleword_api_key="test-key", mode=mode, _env_file=None)


def test_realtime_returns_async_openai():
    from dwp.clients import build_chat_client

    with patch("dwp.clients.settings") as mock_settings:
        mock_settings.mode = "realtime"
        mock_settings.doubleword_api_key = "test-key"
        mock_settings.doubleword_base_url = "https://api.doubleword.ai/v1"

        client = build_chat_client("realtime")

    import openai
    assert isinstance(client, openai.AsyncOpenAI)


def test_async_returns_autobatcher_async():
    from dwp.clients import build_chat_client
    mock_cls = MagicMock(return_value=MagicMock())

    with patch("dwp.clients.settings") as mock_settings, \
         patch.dict("sys.modules", {"autobatcher": MagicMock(AsyncOpenAI=mock_cls)}):
        mock_settings.mode = "async"
        mock_settings.doubleword_api_key = "test-key"
        mock_settings.doubleword_base_url = "https://api.doubleword.ai/v1"

        build_chat_client("async")

    mock_cls.assert_called_once_with(
        api_key="test-key",
        base_url="https://api.doubleword.ai/v1",
    )


def test_batch_returns_autobatcher_batch():
    from dwp.clients import build_chat_client
    mock_cls = MagicMock(return_value=MagicMock())

    with patch("dwp.clients.settings") as mock_settings, \
         patch.dict("sys.modules", {"autobatcher": MagicMock(BatchOpenAI=mock_cls)}):
        mock_settings.mode = "batch"
        mock_settings.doubleword_api_key = "test-key"
        mock_settings.doubleword_base_url = "https://api.doubleword.ai/v1"
        mock_settings.batch_completion_window = "24h"

        build_chat_client("batch")

    mock_cls.assert_called_once_with(
        api_key="test-key",
        base_url="https://api.doubleword.ai/v1",
        completion_window="24h",
    )


def test_invalid_mode_raises():
    from dwp.clients import build_chat_client

    with pytest.raises(ValueError, match="Unknown mode"):
        build_chat_client("turbo")  # type: ignore[arg-type]


# ----- Behavioral tests: each client exposes the call shape agent.py / judges.py rely on. -----


@pytest.mark.parametrize("mode", ["realtime", "async", "batch"])
async def test_client_chat_completions_call_shape(mode):
    """Each mode's client must accept the kwargs `agent.py` and `judges.py` pass:
    `model=`, `messages=`, optionally `response_format=`. The factory is mocked
    at the SDK boundary, so no network is hit."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from dwp.clients import build_chat_client
    from tests.conftest import make_chat_response

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        return_value=make_chat_response('{"ok": true}')
    )
    cls = MagicMock(return_value=fake_client)

    if mode == "realtime":
        ctx = patch("openai.AsyncOpenAI", cls)
    elif mode == "async":
        ctx = patch.dict("sys.modules", {"autobatcher": MagicMock(AsyncOpenAI=cls)})
    else:
        ctx = patch.dict("sys.modules", {"autobatcher": MagicMock(BatchOpenAI=cls)})

    with patch("dwp.clients.settings") as mock_settings, ctx:
        mock_settings.mode = mode
        mock_settings.doubleword_api_key = "test-key"
        mock_settings.doubleword_base_url = "https://api.doubleword.ai/v1"
        mock_settings.batch_completion_window = "24h"

        client = build_chat_client(mode)  # type: ignore[arg-type]
        resp = await client.chat.completions.create(
            model="some/model",
            messages=[{"role": "user", "content": "hi"}],
            response_format={"type": "json_object"},
        )

    assert resp.choices[0].message.content == '{"ok": true}'
    fake_client.chat.completions.create.assert_awaited_once()
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "some/model"
    assert kwargs["messages"][0]["role"] == "user"
    assert kwargs["response_format"] == {"type": "json_object"}
