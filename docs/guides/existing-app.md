# Add Observability to an Existing LLM App

You already have inference. You're missing observability.

Every LLM call your app makes is a black box until you add tracing. Phoenix turns each request into a structured timeline of planning steps, retrieval calls, model invocations, latencies, inputs, and outputs, without touching your business logic.

This guide adds Phoenix tracing to any existing Python app that uses an OpenAI-compatible client. Two additions. No architecture changes.

---

## What you're adding

**01  A tracing bootstrap**: called once at startup. Registers Phoenix as the OpenTelemetry collector.

**02  An auto-instrumentation hook**: one import that wraps every OpenAI SDK call automatically.

That's it. Every `client.chat.completions.create(...)` in your app becomes a span. You don't instrument individual calls.

---

## Step 1: Install

```bash
uv add arize-phoenix-otel openinference-instrumentation-openai
```

---

## Step 2: Bootstrap tracing at startup

Add this before your first LLM call, in `main.py`, `app.py`, or wherever your process initialises:

```python
from phoenix.otel import register

register(
    project_name="my-app",
    auto_instrument=True,   # picks up OpenAI, LangChain, LlamaIndex automatically
    batch=True,             # async span export; no latency on the hot path
)
```

That's the only required change to your application code. `auto_instrument=True` discovers every installed OpenInference instrumentor package and activates them.

---

## Step 3: Start Phoenix

```bash
docker compose -f docker/compose.yaml up -d
# visit http://localhost:6006
```

Set the collector endpoint in `.env` (this is already the default):

```
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
```

---

## Step 4: Switch to Doubleword

If you're currently calling OpenAI directly, replace the base URL. Everything else stays the same:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key="your-doubleword-key",
    base_url="https://api.doubleword.ai/v1",
)
```

Doubleword is OpenAI-compatible. Your existing code, prompts, and tool definitions work without modification. The difference: one key instead of per-provider credentials, and access to async and batch tiers for cost savings.

---

## What you'll see in Phoenix

After a few requests, Phoenix shows:

- One root trace per request
- One child span per LLM call, with model name, token counts, latency, and the full prompt/completion
- A filterable span list: sort by latency, filter by span name, search inputs

From there, add eval scores. [→ Async eval loop](./async-evals.md)
