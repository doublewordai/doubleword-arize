"""One tracing bootstrap. Local Phoenix via Docker."""

from __future__ import annotations

import os

from opentelemetry.sdk.trace import TracerProvider

from .config import settings

_provider: TracerProvider | None = None


def get_tracer_provider() -> TracerProvider:
    """Idempotent. Registers local Phoenix as the OTel collector."""
    global _provider
    if _provider is not None:
        return _provider

    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = settings.phoenix_collector_endpoint

    from phoenix.otel import register

    _provider = register(
        project_name=settings.project_name,
        auto_instrument=True,
        batch=True,
    )
    return _provider
