import json
from pathlib import Path

import pytest

from repopilot.ingest import (
    MAX_CHUNK_CHARS,
    IngestError,
    chunk_lines,
    iter_source_files,
    normalize_github_url,
)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/owner/repo", "https://github.com/owner/repo"),
        ("https://github.com/owner/repo.git", "https://github.com/owner/repo"),
        ("  https://github.com/Owner-1/my.repo_x/  ", "https://github.com/Owner-1/my.repo_x"),
    ],
)
def test_normalize_github_url_accepts_public_repo_urls(url: str, expected: str) -> None:
    assert normalize_github_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "git@github.com:owner/repo.git",
        "http://github.com/owner/repo",
        "https://gitlab.com/owner/repo",
        "https://github.com/owner",
        "https://github.com/owner/repo/tree/main",
        "https://github.com/owner/..",
        "https://github.com/--upload-pack=x/repo",
        "https://github.com.evil.com/owner/repo",
    ],
)
def test_normalize_github_url_rejects_anything_else(url: str) -> None:
    with pytest.raises(IngestError):
        normalize_github_url(url)


def _write(root: Path, rel: str, content: str | bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)


def test_iter_source_files_keeps_code_and_skips_noise(tmp_path: Path) -> None:
    _write(tmp_path, "src/app.py", "print('hi')\n")
    _write(tmp_path, "README.md", "# Title\n")
    _write(tmp_path, "node_modules/lib/index.js", "junk")
    _write(tmp_path, ".git/config", "[core]")
    _write(tmp_path, "image.png", b"\x89PNG\x00\x00")
    _write(tmp_path, "data.py", b"binary\x00content")
    _write(tmp_path, "big.py", "x = 1\n" * 1000)
    _write(tmp_path, "empty.py", "   \n")
    _write(tmp_path, "package-lock.json", "{}")
    (tmp_path / "link.py").symlink_to(tmp_path / "src/app.py")

    files = dict(iter_source_files(tmp_path, max_file_bytes=1000, max_files=100))

    assert files == {"README.md": "# Title\n", "src/app.py": "print('hi')\n"}


def test_iter_source_files_stops_at_max_files(tmp_path: Path) -> None:
    for i in range(5):
        _write(tmp_path, f"f{i}.py", "x = 1\n")

    assert len(list(iter_source_files(tmp_path, max_file_bytes=1000, max_files=3))) == 3


def test_iter_source_files_reads_notebook_cells_not_json(tmp_path: Path) -> None:
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Analysis\n"]},
            {"cell_type": "code", "source": ["import pandas as pd\n", "df = pd.read_csv('x')"]},
        ]
    }
    _write(tmp_path, "nb.ipynb", json.dumps(notebook))

    [(path, text)] = list(iter_source_files(tmp_path, max_file_bytes=10_000, max_files=10))

    assert path == "nb.ipynb"
    assert "import pandas as pd" in text
    assert '"cell_type"' not in text


def test_chunk_lines_tracks_one_based_line_ranges_with_overlap() -> None:
    text = "\n".join(f"line {n}" for n in range(1, 11))

    chunks = chunk_lines("a.py", text, size=4, overlap=1)

    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 4), (4, 7), (7, 10)]
    assert chunks[0].content == "line 1\nline 2\nline 3\nline 4"
    assert all(c.path == "a.py" for c in chunks)


def test_chunk_lines_short_file_is_one_chunk() -> None:
    [chunk] = chunk_lines("a.py", "one\ntwo", size=40, overlap=10)

    assert (chunk.start_line, chunk.end_line) == (1, 2)


def test_chunk_lines_skips_blank_windows_and_empty_text() -> None:
    assert chunk_lines("a.py", "", size=4, overlap=1) == []
    assert chunk_lines("a.py", "\n\n\n\n\n", size=2, overlap=0) == []


def test_chunk_lines_truncates_very_long_lines() -> None:
    [chunk] = chunk_lines("min.js", "x" * (MAX_CHUNK_CHARS * 2), size=40, overlap=10)

    assert len(chunk.content) == MAX_CHUNK_CHARS


@pytest.mark.parametrize(("size", "overlap"), [(4, 4), (4, -1), (0, 0)])
def test_chunk_lines_rejects_bad_window(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_lines("a.py", "x", size=size, overlap=overlap)
