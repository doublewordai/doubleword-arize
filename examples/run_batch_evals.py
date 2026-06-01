"""Batch LLM-as-judge via autobatcher.BatchOpenAI (50-75% off, up to 24h SLA).

Same judge, same spans as run_async_evals.py. The only difference is which
autobatcher client handles the requests.

    MODE=batch uv run python examples/run_batch_evals.py

Set BATCH_COMPLETION_WINDOW=1h in .env for the express lane (dev/CI).
"""

from __future__ import annotations

import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from wasabi import msg

from dwp.config import settings
from dwp.evals.batch import run_batch_evals
from dwp.tracing import get_tracer_provider


async def main() -> None:
    tp = get_tracer_provider()
    msg.info(
        f"Batch eval — autobatcher.BatchOpenAI "
        f"(completion_window={settings.batch_completion_window}, 50-75% off realtime)..."
    )
    result = await run_batch_evals(lookback_hours=24)
    tp.force_flush()
    if not result.empty:
        msg.good(
            f"Logged {len(result)} batch evaluations to Phoenix → {settings.phoenix_collector_endpoint}"
        )
    else:
        msg.warn("No answer spans found. Run run_agent.py first to generate traces.")


if __name__ == "__main__":
    asyncio.run(main())
