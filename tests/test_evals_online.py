"""Online eval loop tests. Phoenix client, judge calls, and the autobatcher
chat client are fully mocked — no network, no Phoenix server needed."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from dwp.evals.online import run_online_evals


def _async_ctx_client() -> MagicMock:
    """Return a MagicMock that behaves as an async context manager (`async with`)."""
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


async def test_empty_dataframe_returns_early():
    with patch("dwp.evals.online._fetch_answer_spans", return_value=pd.DataFrame()):
        result = await run_online_evals(lookback_hours=1)
    assert result.empty


async def test_none_dataframe_returns_early():
    with patch("dwp.evals.online._fetch_answer_spans", return_value=None):
        result = await run_online_evals(lookback_hours=1)
    assert result.empty


async def test_scores_logged_to_phoenix(mock_span_df):
    from dwp.evals.judges import Score

    mock_score = Score(relevance=0.9, hallucination_risk=0.1, tone=0.85, rationale="good")
    chat_client = _async_ctx_client()
    px_client = MagicMock()

    with (
        patch("dwp.evals.online._fetch_answer_spans", return_value=mock_span_df),
        patch("dwp.evals.online.judge", AsyncMock(return_value=mock_score)),
        patch("dwp.evals.online.build_chat_client", return_value=chat_client),
        patch("dwp.evals.online.Client", return_value=px_client),
    ):
        result = await run_online_evals(lookback_hours=1)

    assert not result.empty
    assert len(result) == 2
    # Annotations sent via new Phoenix client API.
    px_client.spans.log_span_annotations_dataframe.assert_called_once()
    kwargs = px_client.spans.log_span_annotations_dataframe.call_args.kwargs
    assert kwargs["annotation_name"] == "quality"
    assert kwargs["annotator_kind"] == "LLM"


async def test_async_client_lifecycle(mock_span_df):
    """Regression guard: the autobatcher client must be entered AND exited.
    Without `async with`, queued judge calls get silently dropped on process exit."""
    from dwp.evals.judges import Score

    mock_score = Score(relevance=0.9, hallucination_risk=0.1, tone=0.85)
    chat_client = _async_ctx_client()

    with (
        patch("dwp.evals.online._fetch_answer_spans", return_value=mock_span_df),
        patch("dwp.evals.online.judge", AsyncMock(return_value=mock_score)),
        patch("dwp.evals.online.build_chat_client", return_value=chat_client),
        patch("dwp.evals.online.Client", return_value=MagicMock()),
    ):
        await run_online_evals(lookback_hours=1)

    chat_client.__aenter__.assert_awaited_once()
    chat_client.__aexit__.assert_awaited_once()


async def test_failed_judge_excluded_from_output(mock_span_df):
    async def flaky_judge(client, q, a):
        if "cheaper" in a:
            raise RuntimeError("boom")
        from dwp.evals.judges import Score
        return Score(relevance=0.9, hallucination_risk=0.1, tone=0.85)

    chat_client = _async_ctx_client()

    with (
        patch("dwp.evals.online._fetch_answer_spans", return_value=mock_span_df),
        patch("dwp.evals.online.judge", side_effect=flaky_judge),
        patch("dwp.evals.online.build_chat_client", return_value=chat_client),
        patch("dwp.evals.online.Client", return_value=MagicMock()),
    ):
        result = await run_online_evals(lookback_hours=1)

    # Only the non-failing row logged.
    assert len(result) == 1
