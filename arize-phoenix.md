# How to Run LLM-as-Judge Evals at Scale with Arize Phoenix and Doubleword

Our goal with this guide is to show you how to connect [Doubleword](https://doubleword.ai) and [Arize Phoenix](https://phoenix.arize.com) - the open-source observability platform - for high throughput yet inexpensive inference and evaluations at scale.

Evaluation is the only way to keep agentic workflows from regressing, but running a frontier model for [LLM-as-a-judge](https://doubleword.ai/glossary#llm-as-a-judge) synchronously on thousands of production traces is an operational bottleneck. You tie up application code, hit aggressive rate limits, and pay premium real-time inference prices for a background task.

By routing your Phoenix evaluation workloads through Doubleword's [batch API](https://docs.doubleword.ai/inference-api/intro-to-doubleword-inference), you can run top-tier models (like [DeepSeek V4 Pro, Qwen-3.6 and others](https://docs.doubleword.ai/inference-api/models)) as your judge for 4-6x less than real-time API costs, with zero rate-limit throttling.

- Tracing - track every generation and judgement as a trace. Break down complex agents and llm calls into individual steps with 'spans' (individual steps such as generating text, fetching data, and using tools like web_search or send_sms that agents use to access information and perform actions).

- Evaluations - evaluate llm outputs against graded references to maintain quality, reliability and consistency. These are often referred to as 'evals'.

> Note: Doubleword seamlessly fits with OpenAI compatible endpoints.

> Using **Arize AX**, the hosted platform? Follow the [Doubleword × Arize AX guide](./README.md) instead.

## Quickstart

- A Doubleword API key - sign up at [app.doubleword.ai](https://app.doubleword.ai/) and generate a key on the API Keys page.
- Arize Phoenix - run it locally/self-hosted (free, open-source) or use [Phoenix Cloud](https://app.phoenix.arize.com).
- Python 3.11+.

If you are using a coding agent to set up Phoenix and Doubleword, you can use the setup prompts to help you get started faster:
```text
Follow the instructions from https://arize.com/docs/PROMPT.md and ask me questions as needed.
```

```text
Use the documentation from https://doubleword.ai/llms.txt for help with the Doubleword inference API
```

## Configuring Arize Phoenix

Pick whichever fits - both give you the same Phoenix UI for traces, datasets, and evals.

### Option A: Run Phoenix locally (open-source)

```bash
pip install arize-phoenix
phoenix serve            # UI at http://localhost:6006
```

You can also run it with Docker - read more [here](https://arize.com/phoenix/) and see the repo [here](https://github.com/Arize-ai/phoenix).

### Option B: Phoenix Cloud

Create an account or log in at [app.phoenix.arize.com](https://app.phoenix.arize.com) and grab an API key from the dashboard.

> Tip: Always keep API keys secure and never share them publicly.

## Running an Evaluation

```bash
pip install autobatcher openinference-instrumentation-openai
```

Add the Phoenix tracing helper:
```bash
pip install arize-phoenix-otel
```

### Step 1 - Connect Phoenix

Pick the block that matches your setup. `auto_instrument=True` traces your OpenAI-compatible calls automatically.

**Local Phoenix:**

```python
import os
from phoenix.otel import register

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"

register(project_name="llm-judge-evals", auto_instrument=True)
```

**Phoenix Cloud:**

```python
import os
from phoenix.otel import register

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"
os.environ["PHOENIX_API_KEY"] = "YOUR_PHOENIX_API_KEY"

register(project_name="llm-judge-evals", auto_instrument=True)
```

That's the entire Phoenix setup. Everything below is traced automatically.

### Step 2 - Generate answers on Doubleword batch

Switching to batches from realtime is easy. `BatchOpenAI` automatically converts and upgrades them to batches.

> Tip: You can see past and current runs as well as live updates on the batches page of [app.doubleword.ai](https://app.doubleword.ai). Choose a model from the [model catalog](https://docs.doubleword.ai/inference-api/model-pricing). Not sure which? Play around and compare with different models on the [playground](https://console.doubleword.ai/playground).

Here we set up a batch client and generate answers on your eval set. In the next step we use a judge to grade the outputs.
```python
import asyncio
from autobatcher import BatchOpenAI

MODEL = "deepseek-ai/DeepSeek-V4-Pro"  # pick from docs.doubleword.ai/inference-api/model-pricing

questions = [
    "What happens if you eat watermelon seeds?",
    "Why do veins look blue?",
    # ...your eval set
]

async def generate(client, question):
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
    )
    return resp.choices[0].message.content

async def main():
    async with BatchOpenAI(
        api_key="YOUR_DOUBLEWORD_API_KEY",
        base_url="https://api.doubleword.ai/v1",
    ) as client:
        answers = await asyncio.gather(*[generate(client, q) for q in questions])
    return answers

answers = asyncio.run(main())
```

Open your project in Phoenix. On the tracing project, each call is there as a span with its prompt, output, and token counts.

### Step 3 - Judge the Answers

The judge is another batch call that hands the model the question and the answer, asks for scores back as JSON. Reuse the same client so the judgements land in the same Phoenix project.

```python
import json

JUDGE = (
    "Score the answer from 0 to 1 on relevance, truthfulness, and tone. "
    'Reply with JSON only: {"relevance": float, "truthfulness": float, "tone": float}.'
)

async def judge(client, question, answer):
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": JUDGE},
            {"role": "user", "content": f"Question: {question}\nAnswer: {answer}"},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

async def main():
    async with BatchOpenAI(
        api_key="YOUR_DOUBLEWORD_API_KEY",
        base_url="https://api.doubleword.ai/v1",
    ) as client:
        answers = await asyncio.gather(*[generate(client, q) for q in questions])
        scores = await asyncio.gather(
            *[judge(client, q, a) for q, a in zip(questions, answers)]
        )
    return scores
```

## Limitations

### Cost tracking

Phoenix shows token counts per call for batch traffic out of the box. Cost in dollars is a little different:

- For cost observability per batch, use Doubleword. The [app.doubleword.ai](https://app.doubleword.ai/batches) shows in-flight, current, and completed batches with the total cost, and the `dw` CLI gives the same via `dw batches analytics`.

Phoenix offers fantastic telemetry for traces, tokens, and scores. The Doubleword dashboard and CLI are a great source of truth for the actual batch spend.

### Order of Operations

- Most Doubleword batches come back very fast. A batch might be 90%+ complete after 10-15 mins, but the remainder could take longer.
- To ensure you grade every item in a batch, wait for the generation batch to complete before grading.

## Going further

- **Full worked example** - the async-evals workbook runs generate-then-judge over a dataset of 817 items from the [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa) dataset as an evaluation experiment with LLM-as-a-judge for $0.50 total.
- **autobatcher** - the batch client used here, also available for TypeScript: [github - autobatcher](https://github.com/doublewordai/autobatcher).
- **Phoenix** - [docs](https://arize.com/docs/phoenix).
- **Using the hosted platform?** [Integrate Doubleword with Arize AX](./README.md).
