"""Batch eval loop tests. BatchOpenAI is mocked as an async context manager.
No real batch job is submitted; no 24h wait occurs.

To run:
    uv run pytest tests/test_evals_batch.py -v
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from dwp.evals.batch import run_batch_evals


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

    # Mock BatchOpenAI as an async context manager
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    mock_px_client = MagicMock()
    mock_px_client.log_evaluations = MagicMock()

    with (
        patch("dwp.evals.batch._fetch_answer_spans", return_value=mock_span_df),
        patch("dwp.evals.batch.judge", AsyncMock(return_value=mock_score)),
        patch("dwp.evals.batch.build_chat_client", return_value=mock_client),
        patch("dwp.evals.batch.px") as mock_px_module,
    ):
        mock_px_module.Client.return_value = mock_px_client
        result = await run_batch_evals(lookback_hours=1)

    assert not result.empty
    assert len(result) == 2
    mock_px_client.log_evaluations.assert_called_once()
    call_arg = mock_px_client.log_evaluations.call_args[0][0]
    assert call_arg.eval_name == "quality_batch"


async def test_low_relevance_labelled_correctly(mock_span_df):
    from dwp.evals.judges import Score

    # relevance < 0.6 → label "low_relevance"
    mock_score = Score(relevance=0.4, hallucination_risk=0.3, tone=0.7, rationale="weak")

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    mock_px_client = MagicMock()

    with (
        patch("dwp.evals.batch._fetch_answer_spans", return_value=mock_span_df),
        patch("dwp.evals.batch.judge", AsyncMock(return_value=mock_score)),
        patch("dwp.evals.batch.build_chat_client", return_value=mock_client),
        patch("dwp.evals.batch.px") as mock_px_module,
    ):
        mock_px_module.Client.return_value = mock_px_client
        result = await run_batch_evals(lookback_hours=1)

    assert (result["label"] == "low_relevance").all()
