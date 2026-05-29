# Build an Async Agent with Doubleword & Phoenix

A typical LLM agent - planning a response, retrieving context, composing an answer - runs in a few seconds. Inference is the expensive part. Doubleword's async tier runs the same requests at **~50% off realtime**. Batch mode takes it to **~90% off**, for any job where a few minutes or hours of latency is acceptable.

This guide builds a working async search-and-answer agent from scratch. Every step is visible in Phoenix as a named span. Switching between Doubleword's three inference tiers requires changing one line of code.

---

## The economics

| Tier | SLA | Cost vs realtime | Use case |
|---|---|---|---|
| Realtime | Immediate | Full price | Interactive, latency-critical |
| Async | ~minutes | ~50% off | Background agents, pipelines |
| Batch | Up to 24h | ~90% off | Bulk evals, synthetic data |

Doubleword pricing on DeepSeek V4 Pro: **$1.74 / $3.48** per million tokens realtime → **$0.87 / $1.74** batch. One API key. No upstream provider credentials.

---

## 01  Set up

```bash
cp .env.example .env
# set DOUBLEWORD_API_KEY=your-key
# leave MODE=realtime and BACKEND=local for now
uv sync
docker compose -f docker/compose.yaml up -d
```

Visit `http://localhost:6006`. Phoenix is running.

---

## 02  The agent shape

The agent does three things. Each becomes a Phoenix span.

```
agent.run
  ├─ planning     decide whether to search
  ├─ searching    retrieve relevant documents (skipped when not needed)
  └─ answering    compose the final response
```

This is **span-first design**: the structure comes before the tracing. Every step maps to a span from day one, so the Phoenix timeline is readable without any post-hoc annotation.

The `searching` step uses a small in-memory corpus (`src/dwp/corpus.py`) with four documents about OpenInference, Phoenix, Doubleword, and autobatcher. It is intentionally minimal. In practice you would replace it with a vector store, a web search tool, a database query, or any other tool call — the span wraps whatever the step does, so Phoenix captures it regardless.

The implementation lives in [`src/dwp/agent.py`](../../src/dwp/agent.py). The core function:

```python
async def run(query: Query, client=None) -> AnswerWithCitations:
    client = client or build_chat_client()     # tier chosen by MODE env var
    with tracer.start_as_current_span("agent.run"):
        decision = await _plan(client, query)
        hits = await _search(query) if decision.needs_search else []
        return await _answer(client, query, hits)
```

`build_chat_client()` is the entire tier-switching mechanism. Change `MODE` and the same agent code runs against a different Doubleword tier.

---

## 03  Run one trace

```bash
uv run python examples/run_agent.py "what is OpenInference?"
```

Phoenix shows: one root trace, three child spans, one OpenAI span underneath each LLM call. Inputs and outputs are captured automatically; no manual attribute setting required beyond the three conceptual spans.

---

## 04  Fan out concurrent traces

```bash
uv run python examples/run_concurrent.py
```

Five queries run under `asyncio.gather`. Phoenix shows five root traces, all overlapping on the timeline. This is the value of async + per-request traces: you see which requests were slow, which retrieved context, which hit the free-answer path, all at a glance.

The code is [examples/run_concurrent.py](../../examples/run_concurrent.py). The relevant line:

```python
results = await asyncio.gather(*(run(Query(text=q)) for q in QUERIES))
```

---

## 05  autobatcher: transparent batching

Both the async and batch tiers are powered by [autobatcher](https://pypi.org/project/autobatcher/), a Doubleword-built library that wraps the Batch API behind the same interface as `AsyncOpenAI`.

```python
# realtime: standard openai.AsyncOpenAI
from openai import AsyncOpenAI

# async lane (~50% off, ~1h): autobatcher.AsyncOpenAI
from autobatcher import AsyncOpenAI

# batch lane (~90% off, 24h): autobatcher.BatchOpenAI
from autobatcher import BatchOpenAI
```

You write the same `await client.chat.completions.create(...)` code regardless. autobatcher collects requests over a configurable window, submits them as a batch job to Doubleword, and resolves the futures when the job completes. No polling loop, no JSONL wrangling.

`build_chat_client()` in [`src/dwp/clients.py`](../../src/dwp/clients.py) returns the right client for the active `MODE`. The agent code never sees the distinction.

## 06  Switch inference tiers

Swap `MODE` in `.env`. No code changes.

```bash
# 50% off realtime: background agents, eval loops
MODE=async uv run python examples/run_concurrent.py

# ~90% off realtime: bulk workloads, up to 24h SLA
MODE=batch uv run python examples/run_concurrent.py
```

The agent produces identical output. Phoenix shows the same span trees. The cost is different.

→ [Add LLM-as-judge evals](./async-evals.md)
→ [Switch observability backends](./local-vs-cloud.md)
