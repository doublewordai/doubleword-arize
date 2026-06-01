"""Online (async-tier) eval loop. Pulls recent answer spans, fans out judge
calls under a semaphore, writes scores back as Phoenix span annotations."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from phoenix.client import Client
from phoenix.client.types.spans import SpanQuery
from wasabi import msg

from ..clients import build_chat_client
from ..config import settings
from .judges import Score, judge


def _fetch_answer_spans(lookback_hours: int) -> pd.DataFrame:
    """Read recent `answering` spans from Phoenix into a dataframe."""
    start = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    query = SpanQuery().where("name == 'answering'")
    return Client().spans.get_spans_dataframe(
        query=query,
        start_time=start,
        project_name=settings.project_name,
    )


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

    sem = asyncio.Semaphore(settings.max_concurrency)

    # `async with` flushes the autobatcher queue on exit — without it,
    # judge calls queued just before the script exits get silently dropped.
    async with build_chat_client(mode="async") as client:
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
                "span_id": span_id,
                "score": (score.relevance + (1 - score.hallucination_risk) + score.tone) / 3,
                "label": "ok" if score.relevance >= 0.6 else "low_relevance",
                "explanation": score.rationale,
            }
        )
    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    Client().spans.log_span_annotations_dataframe(
        dataframe=out,
        annotation_name="quality",
        annotator_kind="LLM",
        sync=True,
    )
    msg.good(f"Logged {len(out)} quality annotations to Phoenix.")
    return out
