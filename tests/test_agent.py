"""Agent tests. All LLM calls are mocked — no network required."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dwp.agent import (
    AnswerWithCitations,
    Query,
    _PlanDecision,
    _answer,
    _plan,
    _search,
    run,
)
from tests.conftest import make_chat_response


def _make_async_client(content: str) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=make_chat_response(content))
    return client


async def test_plan_needs_search_true():
    client = _make_async_client('{"needs_search": true, "reason": "needs facts"}')
    with patch("dwp.agent.tracer"):
        decision = await _plan(client, Query(text="what is OpenInference?"))
    assert isinstance(decision, _PlanDecision)
    assert decision.needs_search is True


async def test_plan_needs_search_false():
    client = _make_async_client('{"needs_search": false, "reason": "small talk"}')
    with patch("dwp.agent.tracer"):
        decision = await _plan(client, Query(text="hello"))
    assert decision.needs_search is False


async def test_search_returns_documents():
    with patch("dwp.agent.tracer"):
        docs = await _search(Query(text="OpenInference tracing"))
    assert len(docs) >= 1
    assert docs[0].id == "doc-openinference"


async def test_answer_with_hits():
    from dwp.corpus import search as corpus_search
    hits = corpus_search("OpenInference", k=2)
    client = _make_async_client("OpenInference provides AI tracing on top of OpenTelemetry.")

    with patch("dwp.agent.tracer"):
        result = await _answer(client, Query(text="what is OpenInference?"), hits)

    assert isinstance(result, AnswerWithCitations)
    assert result.answer
    assert len(result.citations) == len(hits)


async def test_answer_without_hits():
    client = _make_async_client("I don't know.")

    with patch("dwp.agent.tracer"):
        result = await _answer(client, Query(text="hello"), [])

    assert isinstance(result, AnswerWithCitations)
    assert result.citations == []


async def test_run_full_flow():
    plan_resp = make_chat_response('{"needs_search": true, "reason": "needs info"}')
    answer_resp = make_chat_response("OpenInference is an OTel extension for AI.")

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=[plan_resp, answer_resp])

    with patch("dwp.agent.tracer"), patch("dwp.agent.build_chat_client", return_value=client):
        result = await run(Query(text="what is OpenInference?"))

    assert isinstance(result, AnswerWithCitations)
    assert result.answer


async def test_run_no_search_path():
    plan_resp = make_chat_response('{"needs_search": false, "reason": "greeting"}')
    answer_resp = make_chat_response("Hello!")

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=[plan_resp, answer_resp])

    with patch("dwp.agent.tracer"), patch("dwp.agent.build_chat_client", return_value=client):
        result = await run(Query(text="hi"))

    assert result.citations == []
