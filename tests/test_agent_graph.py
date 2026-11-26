from langchain_core.messages import AIMessage, ToolMessage

from repopilot.agent.graph import RepoAgent
from tests.conftest import HashEmbedder
from tests.fakes import CallLog, ScriptedChatModel, tool_call


class FilesStore:
    """Just enough store for the agent tools, without a database."""

    files = {"src/auth.py": "def login(user, password):\n    return check(user, password)\n"}

    def search(self, repo_id, query_text, query_vector, k, mode="hybrid"):
        return [
            {
                "path": "src/auth.py",
                "start_line": 1,
                "end_line": 2,
                "content": self.files["src/auth.py"],
                "score": 0.5,
            }
        ]

    def list_paths(self, repo_id, prefix=""):
        return [p for p in self.files if p.startswith(prefix)]

    def read_file(self, repo_id, path):
        return self.files.get(path)


def _agent(responses: list[AIMessage], log: CallLog, max_tool_rounds: int = 4) -> RepoAgent:
    llm = ScriptedChatModel(responses=responses, log=log)
    return RepoAgent(llm, FilesStore(), HashEmbedder(), max_tool_rounds=max_tool_rounds)


def test_agent_uses_tools_then_answers_with_evidence_and_steps() -> None:
    log = CallLog()
    agent = _agent(
        [
            tool_call("search_code", query="login password"),
            tool_call("read_file", call_id="call-2", path="src/auth.py", start_line=1, end_line=2),
            AIMessage(content="login calls check() [src/auth.py:1-2]"),
        ],
        log,
    )

    result = agent.run(repo_id=1, task="How does login work?")

    assert result.answer == "login calls check() [src/auth.py:1-2]"
    assert [s["tool"] for s in result.steps] == ["search_code", "read_file"]
    assert result.steps[1]["args"] == {"path": "src/auth.py", "start_line": 1, "end_line": 2}
    assert result.sources == ("src/auth.py:1-2",)  # same span from both tools, listed once
    assert "How does login work?" in log.calls[0]["messages"][-1].content


def test_agent_is_forced_to_answer_when_the_tool_budget_runs_out() -> None:
    log = CallLog()
    agent = _agent(
        [
            tool_call("search_code", query="a"),
            tool_call("search_code", call_id="call-2", query="b"),
            AIMessage(content="final answer from evidence"),
        ],
        log,
        max_tool_rounds=2,
    )

    result = agent.run(repo_id=1, task="question")

    assert result.answer == "final answer from evidence"
    assert len(result.steps) == 2
    assert log.calls[-1]["bind_kwargs"].get("tool_choice") == "none"
    assert all(c["bind_kwargs"].get("tool_choice") != "none" for c in log.calls[:-1])


def test_tool_errors_are_returned_to_the_model_instead_of_crashing() -> None:
    log = CallLog()
    agent = _agent(
        [
            tool_call("read_file", path="does/not/exist.py"),
            AIMessage(content="could not find it"),
        ],
        log,
    )

    result = agent.run(repo_id=1, task="question")

    tool_messages = [m for m in log.calls[-1]["messages"] if isinstance(m, ToolMessage)]
    assert "not found" in tool_messages[0].content
    assert result.answer == "could not find it"
    assert result.sources == ()
