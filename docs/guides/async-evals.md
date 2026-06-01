# LLM-as-Judge Evals: Async and Batch Lanes

Running evaluations on the hot path is expensive and adds latency. Don't.

The right approach is to score model outputs off the critical path, after they've been traced. Phoenix stores your spans. A separate eval job reads them, fans out judge calls, and writes scores back as annotations. The agent never knows the eval happened.

Doubleword has two tiers purpose-built for this, both accessed through **autobatcher**, an open-source drop-in replacement for `AsyncOpenAI` that handles batching transparently.

| Lane | Client | Cost vs realtime | Best for |
|---|---|---|---|
| Async | `autobatcher.AsyncOpenAI` | 25-50% off | High-throughput inference where realtime latency isn't required |
| Batch | `autobatcher.BatchOpenAI` | 50-75% off | Bulk workloads with up to a 24h SLA |

**The punchline**: the two eval loops share the same judge prompt and the same Phoenix logging call. The only difference is one line: which autobatcher client you build.

---

## autobatcher

[autobatcher](https://pypi.org/project/autobatcher/) is a Doubleword-built Python library that makes the Batch API feel exactly like the standard async OpenAI SDK.

```bash
pip install autobatcher
```

You write normal async code. autobatcher collects your requests over a configurable window, submits them as a single batch job, and resolves the futures when the job completes. No manual file uploads, no polling loop, no code restructuring.

### AsyncOpenAI: the high-throughput lane

```python
from autobatcher import AsyncOpenAI

async with AsyncOpenAI(
    api_key="your-doubleword-key",
    base_url="https://api.doubleword.ai/v1",
) as client:
    results = await asyncio.gather(*[
        client.chat.completions.create(model=model, messages=msgs)
        for msgs in batch
    ])
```

### BatchOpenAI: the 24-hour lane

```python
from autobatcher import BatchOpenAI

async with BatchOpenAI(
    api_key="your-doubleword-key",
    base_url="https://api.doubleword.ai/v1",
    completion_window="24h",       # default for BatchOpenAI
) as client:
    results = await asyncio.gather(*[
        client.chat.completions.create(model=model, messages=msgs)
        for msgs in batch
    ])
```

The interface is identical. The difference is the `completion_window`, and with it, the cost.

### Serve mode

autobatcher also ships a local OpenAI-compatible HTTP proxy, useful for tools that don't support async batching natively:

```bash
autobatcher serve \
  --base-url https://api.doubleword.ai/v1 \
  --api-key "$DOUBLEWORD_API_KEY" \
  --completion-window 24h \
  --batch-window 60
```

Any client hitting `http://localhost:8080/v1` goes through the batch tier automatically.

---

## The judge

All evals run through a shared LLM-as-judge in [`src/dwp/evals/judges.py`](../../src/dwp/evals/judges.py). It scores each answer on three axes:

| Metric | What it measures |
|---|---|
| `relevance` | Did the answer address the question? |
| `hallucination_risk` | How likely is the answer to contain unsupported claims? |
| `tone` | Is the answer clear and professional? |

Scores are 0.0–1.0. Results are Pydantic-validated `Score` objects before they touch Phoenix.

---

## Online eval loop (autobatcher.AsyncOpenAI, 25-50% off)

```bash
uv run python examples/run_async_evals.py
```

**01  Fetch** recent `answering` spans from Phoenix via `Client().spans.get_spans_dataframe(...)`

**02  Score** each span concurrently; judge calls fan out under an `asyncio.Semaphore`

**03  Log** scores back via `Client().spans.log_span_annotations_dataframe(annotation_name="quality", ...)`; annotations appear on the original spans in Phoenix

Filter by `eval.quality.label == 'low_relevance'` to find answers worth reviewing.

Implementation: [`src/dwp/evals/online.py`](../../src/dwp/evals/online.py)

---

## Batch eval loop (autobatcher.BatchOpenAI, 50-75% off)

```bash
uv run python examples/run_batch_evals.py
```

Same spans, same judge, different client. autobatcher collects all judge calls inside its `async with` block and submits them as a single batch job. Scores attach back to spans as `quality_batch` annotations.

For dev and CI: set `BATCH_COMPLETION_WINDOW=1h` in `.env` to use the 1-hour SLA.

Implementation: [`src/dwp/evals/batch.py`](../../src/dwp/evals/batch.py)

---

## When to prefer each lane

Use **async** when you want high-throughput scoring without paying realtime prices: staging, post-deploy checks, rapid iteration.

Use **batch** when freshness doesn't matter: nightly quality sweeps, bulk evaluation of historical traces, cost-sensitive at scale.

Both can run simultaneously. Phoenix merges `quality` and `quality_batch` annotations on the same spans.
