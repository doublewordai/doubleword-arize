"""Online (async-tier) eval loop. Pulls recent answer spans, fans out judge
calls under a semaphore, writes scores back as Phoenix span evaluations."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import phoenix as px
from phoenix.trace import SpanEvaluations
from wasabi import msg

from ..clients import build_chat_client
from ..config import settings
from .judges import Score, judge


def _fetch_answer_spans(lookback_hours: int) -> pd.DataFrame:
    """Read recent answering spans from Phoenix into a dataframe."""

    start = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    client = px.Client()
    df = client.get_spans_dataframe(
        f"name == 'answering' and start_time >= '{start.isoformat()}'",
        project_name=settings.project_name,
    )
    return df


async def _score_row(
    client: Any, sem: asyncio.Semaphore, span_id: str, query_text: str, answer_text: str
) -> tuple[str, Score | None]:
    async with sem:
        try:
            score = await judge(client, query_text, answer_text)
            return span_id, score
        except Exception as exc:  # surface failures, don't crash the loop
            msg.warn(f"judge failed for span {span_id}: {exc}")
            return span_id, None


def _extract_qa(row: pd.Series) -> tuple[str, str]:
    """Pull query + answer text out of the answering span's attributes."""
    query = row.get("attributes.dwp.query") or ""
    raw = row.get("attributes.output.value") or "{}"
    try:
        answer = json.loads(raw).get("answer", "")
    except json.JSONDecodeError:
        answer = ""
    return str(query), str(answer)


async def run_online_evals(lookback_hours: int = 24) -> pd.DataFrame:
    """End-to-end async eval pass. Returns the dataframe that was logged."""
    df = _fetch_answer_spans(lookback_hours)
    if df is None or df.empty:
        msg.warn("No answering spans found in lookback window.")
        return pd.DataFrame()

    client = build_chat_client(mode="async")
    sem = asyncio.Semaphore(settings.max_concurrency)

    tasks = []
    for span_id, row in df.iterrows():
        q, a = _extract_qa(row)
        if not a:
            continue
        tasks.append(_score_row(client, sem, str(span_id), q, a))

    results = await asyncio.gather(*tasks)

    rows = []
    for span_id, score in results:
        if score is None:
            continue
        rows.append(
            {
                "context.span_id": span_id,
                "score": (score.relevance + (1 - score.hallucination_risk) + score.tone) / 3,
                "label": "ok" if score.relevance >= 0.6 else "low_relevance",
                "explanation": score.rationale,
            }
        )
    out = pd.DataFrame(rows).set_index("context.span_id") if rows else pd.DataFrame()

    if not out.empty:
        px.Client().log_evaluations(SpanEvaluations(eval_name="quality", dataframe=out))
        msg.good(f"Logged {len(out)} quality evaluations to Phoenix.")
    return out
