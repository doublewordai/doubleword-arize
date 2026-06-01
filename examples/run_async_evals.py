"""Online LLM-as-judge eval loop via autobatcher.AsyncOpenAI — Doubleword's
high-throughput inference tier (25-50% off realtime).

Run after generating traces with run_agent.py or run_concurrent.py.

    uv run python examples/run_async_evals.py
"""

from __future__ import annotations

import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from wasabi import msg

from dwp.config import settings
from dwp.evals.online import run_online_evals
from dwp.tracing import get_tracer_provider


async def main() -> None:
    tp = get_tracer_provider()
    msg.info("Scoring recent answer spans — autobatcher.AsyncOpenAI (25-50% off realtime)...")
    result = await run_online_evals(lookback_hours=24)
    tp.force_flush()
    if not result.empty:
        msg.good(f"Logged {len(result)} quality evaluations to Phoenix → {settings.phoenix_collector_endpoint}")
    else:
        msg.warn("No answer spans found. Run run_agent.py first to generate traces.")


if __name__ == "__main__":
    asyncio.run(main())
