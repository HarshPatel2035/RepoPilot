import json
from pathlib import Path

import pytest

from repopilot.evals.dataset import load_eval_set
from repopilot.evals.metrics import Span

DATASET = Path(__file__).parent.parent / "evals" / "itsdangerous.json"


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "set.json"
    path.write_text(json.dumps(payload))
    return path


VALID_QUESTION = {
    "id": "q1",
    "question": "How are signatures verified?",
    "relevant": [{"path": "src/a.py", "start": 10, "end": 20}],
    "reference": "With hmac.compare_digest.",
}


def test_load_eval_set_parses_questions(tmp_path: Path) -> None:
    payload = {"repo": "https://github.com/a/b", "ref": "1.0", "questions": [VALID_QUESTION]}

    eval_set = load_eval_set(_write(tmp_path, payload))

    assert (eval_set.repo, eval_set.ref) == ("https://github.com/a/b", "1.0")
    [question] = eval_set.questions
    assert question.id == "q1"
    assert question.relevant == (Span("src/a.py", 10, 20),)


@pytest.mark.parametrize(
    "bad_question",
    [
        {**VALID_QUESTION, "relevant": []},
        {**VALID_QUESTION, "relevant": [{"path": "a.py", "start": 20, "end": 10}]},
        {**VALID_QUESTION, "question": ""},
        {key: value for key, value in VALID_QUESTION.items() if key != "reference"},
    ],
)
def test_load_eval_set_rejects_bad_questions(tmp_path: Path, bad_question: dict) -> None:
    payload = {"repo": "https://github.com/a/b", "ref": "1.0", "questions": [bad_question]}

    with pytest.raises(ValueError):
        load_eval_set(_write(tmp_path, payload))


def test_load_eval_set_rejects_duplicate_ids(tmp_path: Path) -> None:
    payload = {
        "repo": "https://github.com/a/b",
        "ref": "1.0",
        "questions": [VALID_QUESTION, VALID_QUESTION],
    }

    with pytest.raises(ValueError, match="duplicate"):
        load_eval_set(_write(tmp_path, payload))


def test_shipped_itsdangerous_dataset_is_valid() -> None:
    eval_set = load_eval_set(DATASET)

    assert eval_set.repo == "https://github.com/pallets/itsdangerous"
    assert len(eval_set.questions) >= 20
