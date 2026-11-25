from repopilot.llm import build_messages


def test_build_messages_labels_each_chunk_with_path_and_lines() -> None:
    hits = [
        {"path": "src/auth.py", "start_line": 10, "end_line": 20, "content": "def login(): ..."},
        {"path": "README.md", "start_line": 1, "end_line": 5, "content": "# Project"},
    ]

    messages = build_messages("How does login work?", hits)

    assert messages[0]["role"] == "system"
    assert "cite" in messages[0]["content"].lower()
    user = messages[1]["content"]
    assert "[src/auth.py:10-20]" in user
    assert "def login(): ..." in user
    assert "[README.md:1-5]" in user
    assert user.rstrip().endswith("How does login work?")


def test_build_messages_with_no_hits_says_so() -> None:
    messages = build_messages("Anything?", [])

    assert "no matching code" in messages[1]["content"].lower()
