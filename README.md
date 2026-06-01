<p align="center">
  <a href="https://www.doubleword.ai"><img src="images/doubleword-logo.jpg" height="45" width="auto" alt="Doubleword" /></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://arize.com"><img src="images/arize-logo.jpg" height="28" width="auto" alt="Arize Phoenix" /></a>
</p>

# Async Agent Observability with Doubleword + Arize Phoenix

In this example we run async AI agents and trace them end-to-end with [Arize Phoenix](https://arize.com/phoenix). We use [LLM-as-a-judge](https://doubleword.ai/glossary#llm-as-a-judge) evals with deep at up to 90% less than realtime cost, using [Doubleword](https://www.doubleword.ai) for inference and [autobatcher](https://pypi.org/project/autobatcher/) for transparent batch scheduling. 



### Why This Matters

Agents are not the same as LLM calls. A single user request can trigger a planning step, a retrieval, a tool call, and a final answer, each with its own latency, its own model, and its own failure mode. Standard input/output tracing does not capture this. When something goes wrong, you cannot tell which step caused it.

Debugging means re-running the whole agent, adding logging, and guessing. With span-level tracing you see, per request, whether the model decided to search, what it retrieved, how long each step took, and where the answer degraded. You can run LLM-as-a-judge evals against the *planning* span, not just the final output.

The second problem is cost. Agentic workflows make many model calls per request, so running quality filters and evals at realtime prices is prohibitive at scale. Doubleword's async tier cuts that by ~50%. The batch tier, powered by [autobatcher](https://pypi.org/project/autobatcher/) (a drop-in replacement for `AsyncOpenAI`), cuts it by ~90%, with no code changes beyond swapping one environment variable.

| Tier | Eval cost vs realtime | SLA |
|---|---|---|
| Realtime | Full price | Immediate |
| Async (`autobatcher.AsyncOpenAI`) | ~50% off | ~1 hour |
| Batch (`autobatcher.BatchOpenAI`) | ~90% off | Up to 24 hours |

One Doubleword API key covers all three tiers.

---

## What's in here

A working async search-and-answer agent instrumented from day one with span-first design. Phoenix shows the full trace tree per request and a separate eval loop scores recent agent outputs and attaches scores back to the original spans.

- **Span-first agent design.** Each agent step maps to a Phoenix span. Concurrent requests produce overlapping root traces. You see latency, inputs, and outputs at every level.
- **Two eval lanes, same judge.** Online async evals (~50% off) for fast feedback. Batch evals (~90% off) for nightly sweeps. One line of code apart.
- **Local-first observability.** Traces and eval scores stay on your machine. Phoenix runs in Docker with no cloud account needed.

### Guides

- [Build a new async agent](docs/guides/new-async-agent.md): start here if you're building from scratch
- [Add observability to an existing app](docs/guides/existing-app.md): two additions, no architecture changes
- [Async and batch evals](docs/guides/async-evals.md): LLM-as-judge off the hot path
- [Local Phoenix setup](docs/guides/local-vs-cloud.md): Docker, endpoints, what you see

---

## Cost at a glance

| Tier | How it works | vs realtime |
|---|---|---|
| Realtime | Immediate response | Full price |
| Async | Results in minutes | ~50% off |
| Batch | Up to 24h SLA | ~90% off |

Doubleword pricing on DeepSeek V4 Pro (May 2026): **$1.74 / $3.48** per million tokens realtime → **$0.87 / $1.74** on batch. 

See: [doubleword.ai/pricing](https://doubleword.ai/pricing/) for the latest date models and pricing. 

---

## Quick start

In this project we use the [uv](https://docs.astral.sh/uv/) package manager. Log in to [app.doubleword.ai](https://app.doubleword.ai/) to pick up a `DOUBLEWORD_API_KEY`.

### 1. Configure

```bash
cp .env.example .env
# Set DOUBLEWORD_API_KEY - the only credential you need for inference
```

### 2. Start Phoenix and Postgres

```bash
docker compose -f docker-compose.yaml up -d
# Phoenix UI at http://localhost:6006

# For pheonix only, when bringing to an existing stack:
# docker compose -f compose.pheonix-only.yaml up -d
```

Requires [Docker](https://www.docker.com/get-started/).

### 3. Install

```bash
uv sync --extra dev
```

### 4. Run the agent

```bash
uv run python examples/run_agent.py "what is OpenInference?"
```

Open `http://localhost:6006`. You should see one root trace with `planning`, `searching`, and `answering` spans, plus an OpenAI span under each model call.

### 5. Run concurrent traces

```bash
uv run python examples/run_concurrent.py
```

Five queries run in parallel via `asyncio.gather`. Phoenix shows five overlapping root traces; this is the async-first design in practice.

### 6. Score outputs with evals

First make sure you have some traces from steps 4–5, then:

```bash
# Async lane: results in minutes, ~50% off realtime
MODE=async uv run python examples/run_async_evals.py

# Batch lane: up to 24h, ~90% off realtime
MODE=batch uv run python examples/run_batch_evals.py
```

Phoenix shows `quality` and `quality_batch` annotations on the answer spans. Filter by `eval.quality.label == 'low_relevance'` to find outputs worth reviewing.

---

## Project layout

```
src/dwp/
  config.py          env → validated settings (pydantic-settings)
  tracing.py         one-line bootstrap, local Phoenix
  clients.py         Doubleword client factory: realtime / async / batch
  agent.py           span-first async agent
  corpus.py          in-memory retrieval (swap for your own)
  evals/
    judges.py        Pydantic Score + shared judge prompt
    online.py        async eval loop
    batch.py         24h batch eval loop
examples/            one entry-point per scenario
tests/               pytest suite, all external calls mocked
docker/              local Phoenix (with and without Postgres)
docs/guides/         how-to guides for each part of the stack
```

