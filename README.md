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

This project defaults to **DeepSeek V4 Pro for both the chat agent and the LLM-as-judge**. Running a top-tier model as judge is what makes the batch tier worth using — same model, same prompts, ~90% cheaper. Swap models in `.env` or change the default in [`src/dwp/config.py`](src/dwp/config.py).

See: [doubleword.ai/pricing](https://doubleword.ai/pricing/) for the latest date models and pricing. 

---

## Quick start

In this project we use the [uv](https://docs.astral.sh/uv/) package manager. Log in to [app.doubleword.ai](https://app.doubleword.ai/) to pick up a `DOUBLEWORD_API_KEY`.

### 1. Configure

Copy the example env file and set your Doubleword key:

```bash
# macOS / Linux
cp .env.example .env
```

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

Open `.env` and replace the `DOUBLEWORD_API_KEY=dw-...` placeholder with the key from [app.doubleword.ai](https://app.doubleword.ai/).

All other settings — model names, project name, Phoenix endpoint, concurrency, batch SLA — have working defaults baked into [`src/dwp/config.py`](src/dwp/config.py). You only need to add a setting to your `.env` if you want to override one. `.env` wins over `config.py` defaults.

### 2. Start Phoenix and Postgres

```bash
docker compose -f docker-compose.yaml up -d
# Phoenix UI at http://localhost:6006

# Phoenix only (no Postgres) — use this when bringing it into an existing stack:
# docker compose -f compose.phoenix-only.yaml up -d
```

Requires [Docker](https://www.docker.com/get-started/).

> **Note:** Phoenix ships with an empty project named `default`. If you open the UI now, that's what you'll see — that's expected. The agent writes to its own project (`doubleword-arize`, set via `PROJECT_NAME`), which is created automatically the first time you run step 4.

### 3. Install

```bash
uv sync --extra dev
```

### 4. Run the agent

```bash
uv run python examples/run_agent.py "what is OpenInference?"
```

On success the script prints `Trace sent to Phoenix → http://localhost:6006`. To see the trace:

1. Open [http://localhost:6006/projects](http://localhost:6006/projects) (or click **Projects** in the left nav).
2. Click into the **doubleword-arize** project — *not* `default`, which stays empty.
3. You'll see one root trace `agent.run` with `planning`, (optional) `searching`, and `answering` child spans, plus a `ChatCompletion` LLM span under each model call.

Quick sanity check from the CLI:

```bash
curl http://localhost:6006/v1/projects
# Expect: a project named "doubleword-arize" in the response.
```

### 5. Run concurrent traces

```bash
uv run python examples/run_concurrent.py
```

Five queries run in parallel via `asyncio.gather`. In the **doubleword-arize** project you'll see five overlapping root traces on the timeline — the async-first design in practice.

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

