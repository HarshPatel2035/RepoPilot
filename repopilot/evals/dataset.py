"""Golden question sets: questions about a repo pinned to one tag, with labelled answers."""

import json
from dataclasses import dataclass
from pathlib import Path

from repopilot.evals.metrics import Span


@dataclass(frozen=True)
class GoldQuestion:
    id: str
    question: str
    relevant: tuple[Span, ...]  # where in the repo the answer lives
    reference: str  # a short correct answer, used by the LLM judge


@dataclass(frozen=True)
class EvalSet:
    repo: str
    ref: str  # pinned tag, so line numbers in `relevant` never drift
    questions: tuple[GoldQuestion, ...]


def load_eval_set(path: Path) -> EvalSet:
    """Load and validate a golden set. Raises ValueError naming the bad question."""
    data = json.loads(path.read_text())
    questions = tuple(_parse_question(q) for q in data["questions"])
    ids = [q.id for q in questions]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate question ids: {duplicates}")
    return EvalSet(repo=str(data["repo"]), ref=str(data["ref"]), questions=questions)


def _parse_question(raw: dict) -> GoldQuestion:
    qid = raw.get("id", "?")
    try:
        spans = tuple(Span(str(r["path"]), int(r["start"]), int(r["end"])) for r in raw["relevant"])
        question = GoldQuestion(str(raw["id"]), str(raw["question"]), spans, str(raw["reference"]))
    except (KeyError, TypeError) as exc:
        raise ValueError(f"question {qid}: missing or malformed field ({exc})") from exc
    if not question.question.strip() or not question.reference.strip():
        raise ValueError(f"question {qid}: question and reference must not be empty")
    if not spans:
        raise ValueError(f"question {qid}: needs at least one relevant span")
    if any(s.start < 1 or s.end < s.start for s in spans):
        raise ValueError(f"question {qid}: spans need 1 <= start <= end")
    return question
