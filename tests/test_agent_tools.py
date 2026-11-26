from repopilot.agent.tools import MAX_READ_LINES, make_tools
from tests.conftest import HashEmbedder
from tests.test_agent_graph import FilesStore

LONG_FILE = "\n".join(f"line {n}" for n in range(1, 401))


def _tools(evidence: list[dict]) -> dict:
    store = FilesStore()
    store.files = {**FilesStore.files, "src/long.py": LONG_FILE, "docs/guide.md": "# Guide\n"}
    return {t.name: t for t in make_tools(store, HashEmbedder(), repo_id=1, evidence=evidence)}


def test_search_code_returns_labelled_chunks_and_records_evidence() -> None:
    evidence: list[dict] = []

    output = _tools(evidence)["search_code"].invoke({"query": "login"})

    assert "[src/auth.py:1-2]" in output
    assert "def login" in output
    assert [(e["path"], e["start_line"], e["end_line"]) for e in evidence] == [
        ("src/auth.py", 1, 2)
    ]


def test_read_file_returns_numbered_lines_for_the_requested_range() -> None:
    evidence: list[dict] = []

    output = _tools(evidence)["read_file"].invoke(
        {"path": "src/long.py", "start_line": 10, "end_line": 12}
    )

    assert output.splitlines()[0] == "[src/long.py:10-12]"
    assert "10: line 10" in output
    assert "12: line 12" in output
    assert "13: line 13" not in output
    assert evidence[0]["start_line"] == 10 and evidence[0]["end_line"] == 12


def test_read_file_caps_the_number_of_lines() -> None:
    output = _tools([])["read_file"].invoke({"path": "src/long.py"})

    assert output.splitlines()[0] == f"[src/long.py:1-{MAX_READ_LINES}]"
    assert "truncated" in output.lower()


def test_read_file_reports_missing_files_and_bad_ranges() -> None:
    tools = _tools([])

    assert "not found" in tools["read_file"].invoke({"path": "nope.py"})
    assert "past the end" in tools["read_file"].invoke({"path": "src/auth.py", "start_line": 50})


def test_list_files_filters_by_prefix() -> None:
    output = _tools([])["list_files"].invoke({"prefix": "src/"})

    assert set(output.splitlines()) == {"src/auth.py", "src/long.py"}
