"""Batch (24h) eval loop. Structurally identical to evals.online — the only
meaningful difference is `build_chat_client(mode="batch")` vs `mode="async"`.
That single swap is what delivers ~90% cost reduction vs realtime.

autobatcher.BatchOpenAI collects every `chat.completions.create` call inside
the `async with` block, submits them as a single batch job, and resolves all
futures when the job completes. No manual file upload or polling needed.
"""

from __future__ import annotations

import asyncio

import pandas as pd
from phoenix.client import Client
from wasabi import msg

from ..clients import build_chat_client
from ..config import settings
from .judges import Score, judge
from .online import _extract_qa, _fetch_answer_spans


async def _score_row(
    client: object,
    sem: asyncio.Semaphore,
    span_id: str,
    query_text: str,
    answer_text: str,
) -> tuple[str, Score | None]:
    async with sem:
        try:
            return span_id, await judge(client, query_text, answer_text)
        except Exception as exc:
            msg.warn(f"judge failed for span {span_id}: {exc}")
            return span_id, None


async def run_batch_evals(lookback_hours: int = 24) -> pd.DataFrame:
    """Score recent answer spans through the Doubleword batch tier (~90% off).

    Results are attached back to the original spans as `quality_batch`
    annotations in Phoenix.
    """
    df = _fetch_answer_spans(lookback_hours)
    if df is None or df.empty:
        msg.warn("No answering spans found in lookback window.")
        return pd.DataFrame()

    rows_to_score = [
        (str(span_id), *_extract_qa(row))
        for span_id, row in df.iterrows()
        if _extract_qa(row)[1]
    ]
    if not rows_to_score:
        msg.warn("No scorable rows found.")
        return pd.DataFrame()

    sem = asyncio.Semaphore(settings.max_concurrency)

    # BatchOpenAI flushes the entire batch when the context manager exits.
    async with build_chat_client(mode="batch") as client:
        results = await asyncio.gather(
            *[_score_row(client, sem, span_id, q, a) for span_id, q, a in rows_to_score]
        )

    output_rows = []
    for span_id, score in results:
        if score is None:
            continue
        output_rows.append(
            {
                "span_id": span_id,
                "score": (score.relevance + (1 - score.hallucination_risk) + score.tone) / 3,
                "label": "ok" if score.relevance >= 0.6 else "low_relevance",
                "explanation": score.rationale,
            }
        )

    if not output_rows:
        msg.warn("Batch returned no parsable results.")
        return pd.DataFrame()

    out = pd.DataFrame(output_rows)
    Client().spans.log_span_annotations_dataframe(
        dataframe=out,
        annotation_name="quality_batch",
        annotator_kind="LLM",
        sync=True,
    )
    msg.good(f"Logged {len(out)} batch annotations to Phoenix.")
    return out
