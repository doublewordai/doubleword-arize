"""Shared fixtures used across the test suite."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest


def make_chat_response(content: str) -> MagicMock:
    """Build a ChatCompletion-shaped object with one choice."""
    choice = SimpleNamespace(message=SimpleNamespace(content=content))
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.fixture()
def mock_chat_response():
    return make_chat_response


SCORE_JSON = '{"relevance": 0.9, "hallucination_risk": 0.1, "tone": 0.85, "rationale": "good"}'


@pytest.fixture()
def mock_score_json() -> str:
    return SCORE_JSON


@pytest.fixture()
def mock_span_df() -> pd.DataFrame:
    """Minimal Phoenix-shaped dataframe: two scorable answering spans."""
    import json

    return pd.DataFrame(
        {
            "attributes.dwp.query": ["what is OpenInference?", "how does batch save money?"],
            "attributes.output.value": [
                json.dumps({"answer": "OpenInference is an OTel extension for AI tracing."}),
                json.dumps({"answer": "Batch tier is 50-75% cheaper than realtime."}),
            ],
        },
        index=["span-001", "span-002"],
    )
