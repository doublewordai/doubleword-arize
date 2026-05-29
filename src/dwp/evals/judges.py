"""Shared judge prompt and Pydantic Score model. The online and batch eval
paths use the same prompt; only the transport differs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..config import settings

JUDGE_SYSTEM = (
    "You score an assistant's answer on three axes from 0.0 to 1.0:\n"
    "- relevance: did the answer address the user's question?\n"
    "- hallucination_risk: how likely is the answer to contain unsupported claims?\n"
    "- tone: is the tone clear and professional?\n"
    "Return JSON only: "
    '{"relevance": float, "hallucination_risk": float, "tone": float, "rationale": str}.'
)


class Score(BaseModel):
    relevance: float = Field(ge=0.0, le=1.0)
    hallucination_risk: float = Field(ge=0.0, le=1.0)
    tone: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


def build_judge_messages(query_text: str, answer_text: str) -> list[dict[str, str]]:
    user = (
        f"User question:\n{query_text}\n\n"
        f"Assistant answer:\n{answer_text}\n\n"
        "Score it now."
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


async def judge(client: Any, query_text: str, answer_text: str) -> Score:
    """One judge call against whatever Doubleword tier the client is wired for."""
    resp = await client.chat.completions.create(
        model=settings.model_judge,
        messages=build_judge_messages(query_text, answer_text),
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    return Score.model_validate_json(raw)
