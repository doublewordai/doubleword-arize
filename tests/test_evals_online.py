"""Online eval loop tests. Phoenix client and judge calls are fully mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from dwp.evals.online import run_online_evals


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

    mock_px_client = MagicMock()
    mock_px_client.log_evaluations = MagicMock()

    with (
        patch("dwp.evals.online._fetch_answer_spans", return_value=mock_span_df),
        patch("dwp.evals.online.judge", AsyncMock(return_value=mock_score)),
        patch("dwp.evals.online.build_chat_client", return_value=MagicMock()),
        patch("dwp.evals.online.px") as mock_px_module,
    ):
        mock_px_module.Client.return_value = mock_px_client
        result = await run_online_evals(lookback_hours=1)

    assert not result.empty
    assert len(result) == 2
    mock_px_client.log_evaluations.assert_called_once()
    call_kwargs = mock_px_client.log_evaluations.call_args[0][0]
    assert call_kwargs.eval_name == "quality"


async def test_failed_judge_excluded_from_output(mock_span_df):
    async def flaky_judge(client, q, a):
        if "cheaper" in a:
            raise RuntimeError("boom")
        from dwp.evals.judges import Score
        return Score(relevance=0.9, hallucination_risk=0.1, tone=0.85)

    mock_px_client = MagicMock()

    with (
        patch("dwp.evals.online._fetch_answer_spans", return_value=mock_span_df),
        patch("dwp.evals.online.judge", side_effect=flaky_judge),
        patch("dwp.evals.online.build_chat_client", return_value=MagicMock()),
        patch("dwp.evals.online.px") as mock_px_module,
    ):
        mock_px_module.Client.return_value = mock_px_client
        result = await run_online_evals(lookback_hours=1)

    # Only the non-failing row logged
    assert len(result) == 1
