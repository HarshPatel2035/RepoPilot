import json
import shutil
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from repopilot import indexer
from repopilot.agent.graph import RepoAgent
from repopilot.config import Settings
from repopilot.evals.dataset import EvalSet, GoldQuestion
from repopilot.evals.judge import Judgement
from repopilot.evals.metrics import Span
from repopilot.evals.runner import (
    agent_pipeline,
    ensure_indexed,
    evaluate_generation,
    evaluate_retrieval,
    rag_pipeline,
    report_to_markdown,
)
from repopilot.store import Store
from tests.conftest import TEST_DATABASE_URL, FakeAnswerer, HashEmbedder, requires_db
from tests.fakes import ScriptedChatModel, tool_call

pytestmark = requires_db

EVAL_SET = EvalSet(
    repo="https://github.com/acme/sample",
    ref="v1",
    questions=(
        GoldQuestion(
            "login",
            "how does login check the password",
            (Span("pkg/auth.py", 1, 2),),
            "It calls check_password.",
        ),
        GoldQuestion(
            "cache", "what does the LRU cache evict", (Span("pkg/cache.py", 1, 2),), "Old items."
        ),
    ),
)


@pytest.fixture
def repo_id(store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> int:
    root = tmp_path / "sample"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "auth.py").write_text("def login(user, password):\n    check_password()\n")
    (root / "pkg" / "cache.py").write_text("class LRUCache:\n    evict least recently used\n")
    (root / "README.md").write_text("# Sample project\n")
    cloned: list[str | None] = []

    def fake_clone(url: str, dest: Path, ref: str | None = None) -> None:
        cloned.append(ref)
        shutil.copytree(root, dest)

    monkeypatch.setattr(indexer, "clone_repo", fake_clone)
    settings = Settings(database_url=TEST_DATABASE_URL)

    first = ensure_indexed(store, HashEmbedder(), settings, EVAL_SET)
    second = ensure_indexed(store, HashEmbedder(), settings, EVAL_SET)

    assert first == second
    assert cloned == ["v1"]  # second call reuses the ready index
    return first


def test_evaluate_retrieval_scores_every_question(store: Store, repo_id: int) -> None:
    report = evaluate_retrieval(store, HashEmbedder(), repo_id, EVAL_SET, k=2, mode="hybrid")

    assert report.mode == "hybrid"
    assert report.hit_rate == 1.0
    assert report.mrr == 1.0
    assert [row["id"] for row in report.rows] == ["login", "cache"]
    assert report.rows[0]["retrieved"][0] == "pkg/auth.py:1-2"
    assert "| hybrid |" in report_to_markdown([report])
    json.dumps(report.to_dict())  # must be serialisable for the results file


class FixedJudge:
    def judge(self, question: str, hits: list[dict], answer: str, reference: str) -> Judgement:
        return Judgement(faithfulness=4, correctness=2, reason="ok")


class CitingAnswerer(FakeAnswerer):
    def answer(self, question: str, hits: list[dict]) -> str:
        top = hits[0]
        return f"See [{top['path']}:{top['start_line']}-{top['end_line']}] and [fake.py:1-2]"


def test_evaluate_generation_combines_judge_scores_and_citation_checks(
    store: Store, repo_id: int
) -> None:
    pipeline = rag_pipeline(store, HashEmbedder(), repo_id, k=2, answerer=CitingAnswerer())

    report = evaluate_generation(EVAL_SET, pipeline, judge=FixedJudge(), name="rag")

    assert report.pipeline == "rag"
    assert report.faithfulness == 4.0
    assert report.correctness == 2.0
    assert report.citation_precision == 0.5
    assert report.cited_share == 1.0
    assert report.evidence_hit_rate == 1.0
    assert len(report.rows) == 2
    json.dumps(report.to_dict())


def test_agent_pipeline_scores_the_evidence_the_agent_read(store: Store, repo_id: int) -> None:
    llm = ScriptedChatModel(
        responses=[
            tool_call("read_file", path="pkg/auth.py"),
            AIMessage(content="It calls check_password [pkg/auth.py:1-2]"),
            AIMessage(content="I could not find it."),  # second question: no tools used
        ]
    )
    pipeline = agent_pipeline(RepoAgent(llm, store, HashEmbedder()), repo_id)

    report = evaluate_generation(EVAL_SET, pipeline, judge=FixedJudge(), name="agent")

    assert report.pipeline == "agent"
    assert [r["evidence_hit"] for r in report.rows] == [1.0, 0.0]
    assert report.rows[0]["citation_precision"] == 1.0
    assert report.rows[1]["citation_precision"] is None
