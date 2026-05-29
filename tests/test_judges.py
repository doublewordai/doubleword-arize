"""Judge tests. Score validation + prompt construction + async judge call."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from dwp.evals.judges import Score, build_judge_messages, judge
from tests.conftest import SCORE_JSON, make_chat_response


def test_score_valid():
    s = Score(relevance=0.8, hallucination_risk=0.1, tone=0.9, rationale="clear")
    assert s.relevance == 0.8


def test_score_out_of_bounds():
    with pytest.raises(ValidationError):
        Score(relevance=1.5, hallucination_risk=0.1, tone=0.9)


def test_score_negative_out_of_bounds():
    with pytest.raises(ValidationError):
        Score(relevance=-0.1, hallucination_risk=0.1, tone=0.9)


def test_build_judge_messages_contains_query():
    msgs = build_judge_messages("what is async?", "Async means non-blocking.")
    texts = " ".join(m["content"] for m in msgs)
    assert "what is async?" in texts
    assert "Async means non-blocking." in texts


def test_build_judge_messages_has_system_and_user():
    msgs = build_judge_messages("q", "a")
    roles = [m["role"] for m in msgs]
    assert "system" in roles
    assert "user" in roles


async def test_judge_parses_response():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=make_chat_response(SCORE_JSON))

    score = await judge(client, "what is async?", "Async means non-blocking.")

    assert isinstance(score, Score)
    assert score.relevance == 0.9
    assert score.hallucination_risk == 0.1
    assert score.rationale == "good"
