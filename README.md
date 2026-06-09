# How to Run LLM-as-Judge Evals at Scale with Arize and Doubleword

Our goal with this guide is to show you how to connect [Doubleword](https://doubleword.ai) and [Arize](https://arize.com) for high throughput yet inexpensive inference and evaluations at scale.

Evaluation is the only way to keep agentic workflows from regressing, but running a frontier model for [LLM-as-a-judge](https://doubleword.ai/glossary#llm-as-a-judge) synchronously on thousands of production traces is an operational bottleneck. You tie up application code, hit aggressive rate limits, and pay premium real-time inference prices for a background task.

By routing your Arize evaluation workloads through Doubleword's [batch API](https://docs.doubleword.ai/inference-api/intro-to-doubleword-inference), you can run top-tier models (like [DeepSeek V4 Pro, Qwen-3.6 and others](https://docs.doubleword.ai/inference-api/models)) as your judge for 4-6x less than real-time API costs, with zero rate-limit throttling. 

- Tracing - track every generation and judgement as a trace. Break down complex agents and llm calls into individual steps with 'spans' (individual steps such as generating text, fetching data, and using tools like web_search or send_sms that agents use to access information and perform actions). 

- Evaluations - evaluate llm outputs against graded references to maintain quality, reliability and consistency. These are often referred to as 'evals'. 

> Note: Doubleword seamlessly fits with OpenAI compatible endpoints.

> Using **Arize Phoenix**, the open-source alternative to Arize? Follow the [Doubleword × Arize Phoenix guide](./arize-phoenix.md) instead.

## Quickstart

- A Doubleword API key — sign up at [app.doubleword.ai](https://app.doubleword.ai/) and generate a key on the API Keys page. 
- An Arize account — sign up at [Arize AX](https://app.arize.com). 
- Python 3.11+. 

If you are using a coding agent to set up Arize and Doubleword, you can use the setup prompts to help you get started faster:
```text
Follow the instructions from https://arize.com/docs/PROMPT.md and ask me questions as needed.
```

```text
Use the documentation from https://doubleword.ai/llms.txt for help with the Doubleword inference API
```

## Configuring Arize AX

### Step 1: Log in to Arize SaaS and Doubleword and Obtain API Keys

If you don't have one already, create an account or log in to your [Arize workspace](https://app.arize.com). 

Do the same on the [Doubleword console](https://app.doubleword.ai/) and use the sidebar option 'API Keys' to generate a Doubleword API key.

> Tip: Always keep API keys secure and never share them publicly.

### Step 2: Add Doubleword as an AI Provider

To make Doubleword a first-class citizen in your workspace, add it to your provider list so Arize can securely route evaluation prompts to our async endpoints.

1. In Arize, navigate to Settings > [AI Providers](https://app.arize.com/account/ai-providers).
2. Select the **Custom Providers** tab and add a new Custom Model Endpoint.
3. Fill in the form as follows:

| Field | Value |
| --- | --- |
| Integration Name | `Doubleword` (or any name you like) |
| API Format | `OpenAI` |
| API Key | Your Doubleword API key (e.g. `sk-...`) |
| API Base URL | `https://api.doubleword.ai/v1` — include `/v1`; do **not** add `/chat/completions` |
| Extra Headers | Leave empty |
| OpenAI default models | **Off** — we're not using standard OpenAI models (though GPT-OSS models are available from doubleword) |
| Custom Models → Model name | Your Doubleword model(s), e.g. `deepseek-ai/DeepSeek-V4-Pro` (see the [model catalog](https://docs.doubleword.ai/inference-api/model-pricing)) |

4. (Optional) Under **Advanced Settings**, turn on *Supports function calling* if your models use tools. Set the **Authorized Org / Space** to your workspace, click **Test Integration**, then save.

5. Once you have added your Doubleword API key and the model name, the 'Test Integration' button will check that you are all set to then 'Save Integration'. 

> Note: We are currently working with Arize to become a default, one-click provider in this dropdown. Adding custom providers requires admin privileges for your Arize AX workspace.

### Step 3: Select Your Project

In the Arize sidebar, go to **Observe → Tracing Projects**. Pick a name for this run's project — it appears here automatically the first time you send traces (Step 1 sets it via `project_name`), or you can create one up front. This is where your LLM-as-a-judge traces and scores will live.

### Step 4: Get your Space ID and API key

Tracing is wired up in code (next section), and it needs two values from Arize. Open **Settings** and copy your **Space ID** and **API key** — you'll drop them into the setup below (or your `.env`).

## Running an Evaluation

```bash
pip install autobatcher openinference-instrumentation-openai
```

Add the Arize AX tracing helper:
```bash
pip install arize-otel
```

### Step 1 — Connect Arize

**Arize AX** — copy your Space ID and API key from Settings in the Arize app:

```python
from arize.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor

tracer_provider = register(
    space_id="YOUR_SPACE_ID",
    api_key="YOUR_ARIZE_API_KEY",
    project_name="llm-judge-evals", # Leave this or Rename this to your project name
)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
```

That's the entire Arize setup. Everything below is traced automatically.

### Step 2 - Generate answers on Doubleword batch

Switching to batches from realtime is easy. `BatchOpenAI` automatically converts and upgrades them to batches.

> Tip: You can see past and current runs as well as live updates on the batches page of [app.doubleword.ai](https://app.doubleword.ai). Choose a model from the [model catalog](https://docs.doubleword.ai/inference-api/model-pricing). Not sure which? Play around and compare with different models on the [playground](https://console.doubleword.ai/playground).

Here we show how you can set up a batch client and generate answers on your eval set. In the next step we will use a judge to grade the outputs from this step. 
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

Open your project in [Arize](https://app.arize.com/). On the Tracing Projects page, each call is there as a span with its prompt, output, and token counts. 

### Step 3 — Judge the Answers

The judge is another batch call that hands the model the question and the answer, asks for scores back
as JSON. Reuse the same client so the judgements land in the same Arize project.

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
Arize shows token counts per call for batch traffic out of the box. Cost in dollars
is a little different:

- For cost observability per batch, use Doubleword. The [app.doubleword.ai](https://app.doubleword.ai/batches) shows
  in-flight, current, and completed batches with the total cost, and the `dw` CLI gives the same via
  `dw batches analytics`.

Arize offers fantastic telemetry for traces, tokens, and scores. Doubleword dashboard and cli are a great source of truth for the actual batch spend. 

### Order of Operations

- Most Doubleword batches often come back very fast. A batch might 90%+ might be complete after 10-15 mins but the remainder could take longer to complete. 
- To ensure you grade all of the items in a batch, wait for the generation batch to complete before grading.

## Going further

- **Full worked example** — the async-evals workbook runs generate-then-judge over a dataset of 817 items from the [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa) dataset as an evaluation experiment with LLM-as-a-judge for $0.50 total. 
- **autobatcher** — the batch client used here, also available for TypeScript:
  [github - autobatcher](https://github.com/doublewordai/autobatcher).
- **Arize AX** — [docs](https://arize.com/docs/ax).
- **Prefer open-source?** [Integrate Doubleword with Arize Phoenix](./arize-phoenix.md).
