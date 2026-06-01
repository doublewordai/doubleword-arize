"""Batch eval loop tests. BatchOpenAI and Phoenix client are mocked.
No real batch job is submitted; no 24h wait occurs.

To run:
    uv run pytest tests/test_evals_batch.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from dwp.evals.batch import run_batch_evals


def _async_ctx_client() -> MagicMock:
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


async def test_empty_dataframe_returns_early():
    with patch("dwp.evals.batch._fetch_answer_spans", return_value=pd.DataFrame()):
        result = await run_batch_evals(lookback_hours=1)
    assert result.empty


async def test_none_dataframe_returns_early():
    with patch("dwp.evals.batch._fetch_answer_spans", return_value=None):
        result = await run_batch_evals(lookback_hours=1)
    assert result.empty


async def test_batch_scores_logged_to_phoenix(mock_span_df):
    from dwp.evals.judges import Score

    mock_score = Score(relevance=0.9, hallucination_risk=0.1, tone=0.85, rationale="good")
    chat_client = _async_ctx_client()
    px_client = MagicMock()

    with (
        patch("dwp.evals.batch._fetch_answer_spans", return_value=mock_span_df),
        patch("dwp.evals.batch.judge", AsyncMock(return_value=mock_score)),
        patch("dwp.evals.batch.build_chat_client", return_value=chat_client),
        patch("dwp.evals.batch.Client", return_value=px_client),
    ):
        result = await run_batch_evals(lookback_hours=1)

    assert not result.empty
    assert len(result) == 2
    px_client.spans.log_span_annotations_dataframe.assert_called_once()
    kwargs = px_client.spans.log_span_annotations_dataframe.call_args.kwargs
    assert kwargs["annotation_name"] == "quality_batch"
    assert kwargs["annotator_kind"] == "LLM"


async def test_batch_client_lifecycle(mock_span_df):
    """Regression guard: `async with` on BatchOpenAI is required — that's
    what triggers batch submission. Without `__aexit__`, the batch never flushes."""
    from dwp.evals.judges import Score

    mock_score = Score(relevance=0.9, hallucination_risk=0.1, tone=0.85)
    chat_client = _async_ctx_client()

    with (
        patch("dwp.evals.batch._fetch_answer_spans", return_value=mock_span_df),
        patch("dwp.evals.batch.judge", AsyncMock(return_value=mock_score)),
        patch("dwp.evals.batch.build_chat_client", return_value=chat_client),
        patch("dwp.evals.batch.Client", return_value=MagicMock()),
    ):
        await run_batch_evals(lookback_hours=1)

    chat_client.__aenter__.assert_awaited_once()
    chat_client.__aexit__.assert_awaited_once()


async def test_low_relevance_labelled_correctly(mock_span_df):
    from dwp.evals.judges import Score

    mock_score = Score(relevance=0.4, hallucination_risk=0.3, tone=0.7, rationale="weak")
    chat_client = _async_ctx_client()

    with (
        patch("dwp.evals.batch._fetch_answer_spans", return_value=mock_span_df),
        patch("dwp.evals.batch.judge", AsyncMock(return_value=mock_score)),
        patch("dwp.evals.batch.build_chat_client", return_value=chat_client),
        patch("dwp.evals.batch.Client", return_value=MagicMock()),
    ):
        result = await run_batch_evals(lookback_hours=1)

    assert (result["label"] == "low_relevance").all()
