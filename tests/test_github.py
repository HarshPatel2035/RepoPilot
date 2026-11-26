import json

import httpx
import pytest

from repopilot.agent.github import (
    MAX_BODY_CHARS,
    GitHubError,
    IssueNotFound,
    fetch_issue,
    issue_task,
    owner_repo,
)

ISSUE = {
    "number": 7,
    "title": "Crash on empty input",
    "body": "Calling parse('') raises IndexError.",
    "state": "open",
    "html_url": "https://github.com/acme/tool/issues/7",
}
COMMENTS = [{"user": {"login": "maintainer"}, "body": "Can you share the traceback?"}]


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_issue_reads_issue_and_comments() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/comments"):
            return httpx.Response(200, json=COMMENTS)
        return httpx.Response(200, json=ISSUE)

    issue = fetch_issue("acme", "tool", 7, token="t0k", client=_client(handler))

    assert (issue.number, issue.title, issue.state) == (7, "Crash on empty input", "open")
    assert issue.comments == ("maintainer: Can you share the traceback?",)
    assert seen[0].url.path == "/repos/acme/tool/issues/7"
    assert seen[0].headers["authorization"] == "Bearer t0k"


def test_fetch_issue_truncates_long_bodies() -> None:
    long_issue = {**ISSUE, "body": "x" * (MAX_BODY_CHARS * 2)}

    def handler(request: httpx.Request) -> httpx.Response:
        body = [] if request.url.path.endswith("/comments") else long_issue
        return httpx.Response(200, content=json.dumps(body))

    issue = fetch_issue("acme", "tool", 7, client=_client(handler))

    assert len(issue.body) <= MAX_BODY_CHARS + 20


def test_fetch_issue_maps_errors() -> None:
    with pytest.raises(IssueNotFound):
        fetch_issue("acme", "tool", 7, client=_client(lambda r: httpx.Response(404)))
    with pytest.raises(GitHubError):
        fetch_issue("acme", "tool", 7, client=_client(lambda r: httpx.Response(403)))


def test_owner_repo_parses_plain_and_pinned_urls() -> None:
    assert owner_repo("https://github.com/acme/tool") == ("acme", "tool")
    assert owner_repo("https://github.com/acme/tool@v1.0") == ("acme", "tool")


def test_issue_task_marks_issue_text_as_untrusted_data() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[] if request.url.path.endswith("/comments") else ISSUE)

    task = issue_task(fetch_issue("acme", "tool", 7, client=_client(handler)))

    assert "Crash on empty input" in task
    assert "untrusted" in task.lower()
