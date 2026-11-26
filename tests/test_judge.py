import pytest

from repopilot.evals.judge import Judgement, build_judge_messages, parse_judgement


def test_parse_judgement_reads_scores() -> None:
    text = '{"faithfulness": 5, "correctness": 3, "reason": "misses key rotation"}'

    assert parse_judgement(text) == Judgement(5, 3, "misses key rotation")


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"faithfulness": 6, "correctness": 3, "reason": "x"}',
        '{"faithfulness": 0, "correctness": 3, "reason": "x"}',
        '{"faithfulness": "5", "correctness": 3, "reason": "x"}',
        '{"correctness": 3, "reason": "x"}',
        "[1, 2]",
    ],
)
def test_parse_judgement_rejects_malformed_output(text: str) -> None:
    with pytest.raises(ValueError):
        parse_judgement(text)


def test_judge_prompt_contains_everything_the_judge_needs() -> None:
    hits = [{"path": "src/a.py", "start_line": 1, "end_line": 9, "content": "def f(): ..."}]

    messages = build_judge_messages("What is f?", hits, "f does X [src/a.py:1-9]", "f does X")

    prompt = messages[-1]["content"]
    for expected in ("What is f?", "def f(): ...", "f does X [src/a.py:1-9]", "Reference answer"):
        assert expected in prompt
