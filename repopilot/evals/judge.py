"""LLM-as-judge: score an answer for faithfulness to the context and correctness."""

import json
from dataclasses import dataclass
from typing import Any

from repopilot.llm import build_messages

JUDGE_SYSTEM = """You grade answers produced by a code question-answering system.
Return ONLY a JSON object: {"faithfulness": int, "correctness": int, "reason": str}

faithfulness (1-5): is every claim in the answer supported by the context chunks?
  5 = fully supported, 3 = some unsupported details, 1 = mostly invented.
correctness (1-5): does the answer agree with the reference answer?
  5 = same facts, 3 = partly right or missing key facts, 1 = wrong or no answer.
"Could not find it" scores correctness 1 unless the reference also says it is not there.
reason: one sentence explaining the lowest score."""


@dataclass(frozen=True)
class Judgement:
    faithfulness: int
    correctness: int
    reason: str


def build_judge_messages(
    question: str, hits: list[dict[str, Any]], answer: str, reference: str
) -> list[dict[str, str]]:
    # Reuse the exact context the answering model saw.
    context = build_messages(question, hits)[1]["content"]
    user = (
        f"{context}\n\nAnswer to grade:\n{answer}\n\n"
        f"Reference answer:\n{reference}\n\nReturn the JSON object now."
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def parse_judgement(text: str) -> Judgement:
    """Validate the judge's JSON. Raises ValueError rather than guessing a score."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge did not return JSON: {text[:200]!r}") from exc
    if not isinstance(data, dict):
        raise ValueError("judge output must be a JSON object")
    scores = {}
    for name in ("faithfulness", "correctness"):
        value = data.get(name)
        if type(value) is not int or not 1 <= value <= 5:
            raise ValueError(f"judge {name} must be an int from 1 to 5, got {value!r}")
        scores[name] = value
    return Judgement(scores["faithfulness"], scores["correctness"], str(data.get("reason", "")))


class GroqJudge:
    def __init__(self, api_key: str, model: str, timeout_seconds: float = 60.0) -> None:
        from groq import Groq

        self._client = Groq(api_key=api_key, timeout=timeout_seconds)
        self._model = model

    def judge(
        self, question: str, hits: list[dict[str, Any]], answer: str, reference: str
    ) -> Judgement:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=build_judge_messages(question, hits, answer, reference),
            temperature=0,
            response_format={"type": "json_object"},
        )
        return parse_judgement(response.choices[0].message.content or "")
