import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from repopilot import indexer
from repopilot.api import create_app
from repopilot.config import Settings
from repopilot.ingest import IngestError
from repopilot.store import Store
from tests.conftest import TEST_DATABASE_URL, FakeAnswerer, HashEmbedder, requires_db

pytestmark = requires_db

REPO_URL = "https://github.com/acme/sample"


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    root = tmp_path / "sample"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "auth.py").write_text(
        "def login(user, password):\n    return check_password(user, password)\n"
    )
    (root / "README.md").write_text("# Sample\nA small demo project.\n")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "x.js").write_text("ignored")
    return root


@pytest.fixture
def answerer() -> FakeAnswerer:
    return FakeAnswerer()


def _client(store: Store, answerer: FakeAnswerer | None) -> TestClient:
    settings = Settings(database_url=TEST_DATABASE_URL, groq_api_key="")
    app = create_app(settings, store=store, embedder=HashEmbedder(), answerer=answerer)
    return TestClient(app)


@pytest.fixture
def client(
    store: Store, answerer: FakeAnswerer, sample_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setattr(
        indexer, "clone_repo", lambda url, dest, ref=None: shutil.copytree(sample_repo, dest)
    )
    with _client(store, answerer) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_then_ask_returns_answer_with_sources(
    client: TestClient, answerer: FakeAnswerer
) -> None:
    created = client.post("/repos", json={"url": REPO_URL + ".git"})
    assert created.status_code == 202
    repo_id = created.json()["id"]
    assert created.json()["url"] == REPO_URL

    repo = client.get(f"/repos/{repo_id}").json()
    assert repo["status"] == "ready"
    assert repo["chunk_count"] == 2  # auth.py + README.md; node_modules skipped

    response = client.post(
        f"/repos/{repo_id}/ask", json={"question": "how does login check the password?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "fake answer"
    assert body["sources"][0]["path"] == "pkg/auth.py"
    assert body["sources"][0]["start_line"] == 1
    question, hits = answerer.calls[0]
    assert question == "how does login check the password?"
    assert hits[0]["content"].startswith("def login")


def test_list_repos(client: TestClient) -> None:
    client.post("/repos", json={"url": REPO_URL})

    response = client.get("/repos")

    assert [r["url"] for r in response.json()] == [REPO_URL]


@pytest.mark.parametrize("url", ["file:///etc/passwd", "https://gitlab.com/a/b", "not a url"])
def test_add_repo_rejects_non_github_urls(client: TestClient, url: str) -> None:
    assert client.post("/repos", json={"url": url}).status_code == 422


def test_failed_clone_marks_repo_failed_and_blocks_questions(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(url: str, dest: Path, ref: str | None = None) -> None:
        raise IngestError("could not clone the repository")

    monkeypatch.setattr(indexer, "clone_repo", fail)
    repo_id = client.post("/repos", json={"url": REPO_URL}).json()["id"]

    repo = client.get(f"/repos/{repo_id}").json()
    ask = client.post(f"/repos/{repo_id}/ask", json={"question": "anything here?"})

    assert (repo["status"], repo["error"]) == ("failed", "could not clone the repository")
    assert ask.status_code == 409


def test_unexpected_indexing_error_is_not_leaked(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def crash(url: str, dest: Path, ref: str | None = None) -> None:
        raise RuntimeError("secret internal path /srv/x")

    monkeypatch.setattr(indexer, "clone_repo", crash)
    repo_id = client.post("/repos", json={"url": REPO_URL}).json()["id"]

    error = client.get(f"/repos/{repo_id}").json()["error"]

    assert "secret" not in error


def test_unknown_repo_returns_404(client: TestClient) -> None:
    assert client.get("/repos/999").status_code == 404
    assert client.post("/repos/999/ask", json={"question": "hello?"}).status_code == 404


def test_ask_validates_question_length(client: TestClient) -> None:
    repo_id = client.post("/repos", json={"url": REPO_URL}).json()["id"]

    assert client.post(f"/repos/{repo_id}/ask", json={"question": ""}).status_code == 422
    too_long = {"question": "x" * 2001}
    assert client.post(f"/repos/{repo_id}/ask", json=too_long).status_code == 422


def test_ask_without_llm_key_returns_503(
    store: Store, sample_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        indexer, "clone_repo", lambda url, dest, ref=None: shutil.copytree(sample_repo, dest)
    )
    with _client(store, answerer=None) as no_llm:
        repo_id = no_llm.post("/repos", json={"url": REPO_URL}).json()["id"]

        response = no_llm.post(f"/repos/{repo_id}/ask", json={"question": "what is this?"})

    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]
