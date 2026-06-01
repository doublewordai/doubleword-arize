# Local Phoenix with Docker

All traces land in a local Phoenix instance running in Docker. Nothing leaves your machine.

---

## Start Phoenix

```bash
docker compose -f docker-compose.yaml up -d
# Phoenix UI at http://localhost:6006
```

This starts Phoenix with Postgres for persistence across restarts. For a lighter smoke-test with no storage:

```bash
docker compose -f compose.phoenix-only.yaml up -d
```

---

## Configuration

In `.env`:

```
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
```

This is the default; no changes needed if you're running the standard Docker setup.

---

## What Phoenix shows

- One root trace per agent request
- Child spans for each agent step (planning, searching, answering)
- OpenAI spans under each model call: model name, token counts, latency, full prompt/completion
- Eval scores attached to answer spans after running the eval loops

→ [Back to new async agent guide](./new-async-agent.md)
