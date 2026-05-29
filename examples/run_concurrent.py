"""Fan N queries out concurrently via asyncio.gather.

Phoenix shows N overlapping root traces on the timeline — the async-first
design in practice.

    uv run python examples/run_concurrent.py
"""

from __future__ import annotations

import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from wasabi import msg

from dwp.agent import Query, run
from dwp.config import settings
from dwp.tracing import get_tracer_provider

QUERIES = [
    "what is OpenInference?",
    "how does Doubleword's batch tier save money?",
    "what backends can Phoenix talk to?",
    "what does autobatcher do?",
    "how does Phoenix tracing work with async agents?",
]


async def main() -> None:
    tp = get_tracer_provider()
    msg.info(f"Sending {len(QUERIES)} queries concurrently via asyncio.gather...")
    results = await asyncio.gather(*(run(Query(text=q)) for q in QUERIES))
    msg.divider("Results")
    for q, r in zip(QUERIES, results):
        msg.text(f"\nQ: {q}")
        msg.text(f"A: {r.answer[:200]}")
    tp.force_flush()
    msg.good(f"{len(QUERIES)} traces sent to Phoenix → {settings.phoenix_collector_endpoint}")


if __name__ == "__main__":
    asyncio.run(main())
