"""Single query, one trace.

    uv run python examples/run_agent.py "what is OpenInference?"
"""

from __future__ import annotations

import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from wasabi import msg

from dwp.agent import Query, run
from dwp.config import settings
from dwp.tracing import get_tracer_provider


async def main(text: str) -> None:
    tp = get_tracer_provider()
    msg.info(f"Query: {text}")
    result = await run(Query(text=text))
    msg.divider("Answer")
    msg.text(result.answer)
    if result.citations:
        msg.divider("Citations")
        for c in result.citations:
            msg.text(f"  {c.doc_id}  {c.title}")
    tp.force_flush()
    msg.good(f"Trace sent to Phoenix → {settings.phoenix_collector_endpoint}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        msg.fail("No query provided.")
        msg.text("Usage:  uv run python examples/run_agent.py \"your prompt here\"")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
