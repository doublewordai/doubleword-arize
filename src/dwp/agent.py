"""Search-and-answer async agent. Span-first design: the three conceptual steps
(planning / searching / answering) are explicit spans, so Phoenix shows them
without any post-hoc instrumentation. Auto-instrumentation captures the LLM
calls underneath each span."""

from __future__ import annotations

import json
from typing import Any

from opentelemetry import trace
from pydantic import BaseModel, Field

from .clients import build_chat_client
from .config import settings
from .corpus import Document, search

tracer = trace.get_tracer("dwp.agent")


class Query(BaseModel):
    text: str


class Citation(BaseModel):
    doc_id: str
    title: str


class AnswerWithCitations(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class _PlanDecision(BaseModel):
    needs_search: bool
    reason: str


_PLAN_SYSTEM = (
    "You decide whether a user question needs a knowledge-base lookup before answering. "
    "Reply with JSON: {\"needs_search\": bool, \"reason\": str}. "
    "Set needs_search=true unless the question is pure small talk."
)

_ANSWER_SYSTEM_GROUNDED = (
    "Answer the user's question using only the provided context. If the context is "
    "insufficient, say so. Be concise."
)
_ANSWER_SYSTEM_FREE = (
    "Answer the user's question concisely. No external context was retrieved."
)


async def _plan(client: Any, query: Query) -> _PlanDecision:
    with tracer.start_as_current_span("planning") as span:
        span.set_attribute("dwp.query", query.text)
        resp = await client.chat.completions.create(
            model=settings.model_chat,
            messages=[
                {"role": "system", "content": _PLAN_SYSTEM},
                {"role": "user", "content": query.text},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        decision = _PlanDecision.model_validate_json(raw)
        span.set_attribute("dwp.needs_search", decision.needs_search)
        return decision


async def _search(query: Query) -> list[Document]:
    with tracer.start_as_current_span("searching") as span:
        span.set_attribute("dwp.query", query.text)
        hits = search(query.text, k=3)
        span.set_attribute("dwp.hits", len(hits))
        span.set_attribute("dwp.hit_ids", ",".join(d.id for d in hits))
        return hits


async def _answer(
    client: Any, query: Query, hits: list[Document]
) -> AnswerWithCitations:
    with tracer.start_as_current_span("answering") as span:
        span.set_attribute("dwp.query", query.text)
        if hits:
            context = "\n\n".join(f"[{d.id}] {d.title}: {d.text}" for d in hits)
            messages = [
                {"role": "system", "content": _ANSWER_SYSTEM_GROUNDED},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query.text}"},
            ]
        else:
            messages = [
                {"role": "system", "content": _ANSWER_SYSTEM_FREE},
                {"role": "user", "content": query.text},
            ]
        resp = await client.chat.completions.create(
            model=settings.model_chat,
            messages=messages,
        )
        text = resp.choices[0].message.content or ""
        citations = [Citation(doc_id=d.id, title=d.title) for d in hits]
        result = AnswerWithCitations(answer=text, citations=citations)
        span.set_attribute("dwp.answer_chars", len(text))
        # store the answer payload so the eval loop can read it back later
        span.set_attribute("output.value", json.dumps(result.model_dump()))
        return result


async def run(query: Query, client: Any | None = None) -> AnswerWithCitations:
    """One async call, three nested spans, exactly one root trace per invocation."""
    client = client or build_chat_client()
    with tracer.start_as_current_span("agent.run") as span:
        span.set_attribute("dwp.mode", settings.mode)
        span.set_attribute("input.value", query.text)
        decision = await _plan(client, query)
        hits = await _search(query) if decision.needs_search else []
        return await _answer(client, query, hits)
