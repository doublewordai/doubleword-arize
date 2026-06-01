"""A deliberately tiny in-memory corpus. The PoC is about the agent shape and
the tracing/eval pipeline — not retrieval quality. Swap for a real index later."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str


_DOCS: tuple[Document, ...] = (
    Document(
        id="doc-openinference",
        title="OpenInference",
        text=(
            "OpenInference is a set of conventions and plugins complementary to OpenTelemetry "
            "for tracing AI applications. Phoenix is natively built on OpenInference but the "
            "instrumentation works against any OTel-compatible backend."
        ),
    ),
    Document(
        id="doc-phoenix",
        title="Arize Phoenix",
        text=(
            "Phoenix is an open-source LLM observability platform for tracing, evaluation, and "
            "debugging. Self-host via Docker (arizephoenix/phoenix) for local development — "
            "traces and eval scores stay on your machine."
        ),
    ),
    Document(
        id="doc-doubleword",
        title="Doubleword inference",
        text=(
            "Doubleword exposes an OpenAI-compatible API at api.doubleword.ai/v1 with three "
            "tiers: realtime, async (25-50% off), and batch (24h, 50-75% off). One API key covers "
            "all three; upstream provider keys never appear in client code."
        ),
    ),
    Document(
        id="doc-autobatcher",
        title="autobatcher",
        text=(
            "autobatcher is a drop-in replacement for openai.AsyncOpenAI that transparently "
            "batches requests behind the Batch API. Use AsyncOpenAI for async-tier discounts "
            "and BatchOpenAI for full batch pricing."
        ),
    ),
)


def search(query: str, k: int = 3) -> list[Document]:
    """Toy lexical scorer. Returns top-k documents by overlap with the query."""
    q = {t.lower() for t in query.split() if len(t) > 2}
    scored = [
        (sum(1 for w in d.text.lower().split() if w.strip(".,") in q), d) for d in _DOCS
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for score, d in scored[:k] if score > 0] or list(_DOCS[:k])
