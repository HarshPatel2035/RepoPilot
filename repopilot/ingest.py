"""Turn a public GitHub repository into line-numbered text chunks."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

ALLOWED_SUFFIXES = frozenset(
    {
        ".py", ".ipynb", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt",
        ".c", ".h", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".scala", ".sh",
        ".sql", ".md", ".rst", ".txt", ".yml", ".yaml", ".toml", ".json",
    }
)  # fmt: skip
SKIP_DIRS = frozenset(
    {
        ".git", "node_modules", "__pycache__", ".venv", "venv", "env", "dist", "build",
        ".next", "vendor", "target", ".mypy_cache", ".pytest_cache", ".tox", "site-packages",
    }
)  # fmt: skip
SKIP_FILES = frozenset(
    {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "uv.lock", "Cargo.lock"}
)
MAX_CHUNK_CHARS = 4_000
CLONE_TIMEOUT_SECONDS = 180

# Owner: letters, digits, hyphens (GitHub's rule). Repo: letters, digits, ".", "_", "-".
_GITHUB_URL = re.compile(
    r"^https://github\.com/([A-Za-z0-9][A-Za-z0-9-]*)/([A-Za-z0-9._-]+?)(?:\.git)?/?$"
)
_GIT_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,99}$")


class IngestError(Exception):
    """A problem with the requested repository. The message is safe to show to users."""


@dataclass(frozen=True)
class Chunk:
    path: str
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    content: str


def normalize_github_url(url: str) -> str:
    """Return https://github.com/<owner>/<repo>, or raise IngestError for any other URL.

    This is the trust boundary for `git clone`: only public GitHub HTTPS URLs get through,
    which blocks file://, ssh, other hosts and option injection.
    """
    match = _GITHUB_URL.match(url.strip())
    if not match or match.group(2) in {".", ".."}:
        raise IngestError("URL must look like https://github.com/<owner>/<repo>")
    return f"https://github.com/{match.group(1)}/{match.group(2)}"


def clone_repo(url: str, dest: Path, ref: str | None = None) -> None:
    """Shallow-clone a public GitHub repository into `dest`, at a branch or tag if given."""
    # shortcut: no disk-size cap on clones; add one before accepting untrusted public traffic.
    safe_url = normalize_github_url(url)
    branch = [] if ref is None else ["--branch", _safe_ref(ref)]
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}  # private repos fail fast instead of prompting
    command = [
        "git",
        "clone",
        "--depth",
        "1",
        "--single-branch",
        *branch,
        "--",
        safe_url,
        str(dest),
    ]
    try:
        subprocess.run(  # noqa: S603 - fixed argv, URL validated above
            command, check=True, capture_output=True, timeout=CLONE_TIMEOUT_SECONDS, env=env
        )
    except subprocess.TimeoutExpired as exc:
        raise IngestError(f"cloning took longer than {CLONE_TIMEOUT_SECONDS} seconds") from exc
    except subprocess.CalledProcessError as exc:
        raise IngestError(
            "could not clone the repository; is it public and spelled right?"
        ) from exc


def _safe_ref(ref: str) -> str:
    # Must not start with "-", or git would read it as an option.
    if not _GIT_REF.match(ref):
        raise IngestError("ref must be a branch or tag name like 'main' or 'v2.2.0'")
    return ref


def iter_source_files(root: Path, max_file_bytes: int, max_files: int) -> Iterator[tuple[str, str]]:
    """Yield (posix relative path, text) for each indexable source file under `root`."""
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):  # does not follow symlinked dirs
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)  # os.walk prunes in place
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if not _is_candidate(path, max_file_bytes):
                continue
            text = _read_text(path)
            if text is None or not text.strip():
                continue
            yield path.relative_to(root).as_posix(), text
            count += 1
            if count >= max_files:
                return


def chunk_lines(path: str, text: str, size: int = 40, overlap: int = 10) -> list[Chunk]:
    """Split text into overlapping windows of `size` lines, keeping 1-based line numbers."""
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("need size >= 1 and 0 <= overlap < size")
    lines = text.splitlines()
    chunks = []
    for start in range(0, len(lines), size - overlap):
        window = lines[start : start + size]
        if any(line.strip() for line in window):
            content = "\n".join(window)[:MAX_CHUNK_CHARS]  # minified files have huge lines
            chunks.append(Chunk(path, start + 1, start + len(window), content))
        if start + size >= len(lines):
            break
    return chunks


def _is_candidate(path: Path, max_file_bytes: int) -> bool:
    # Symlinks could point outside the clone, so they are never read.
    if path.is_symlink() or path.name in SKIP_FILES:
        return False
    return path.suffix.lower() in ALLOWED_SUFFIXES and path.stat().st_size <= max_file_bytes


def _read_text(path: Path) -> str | None:
    raw = path.read_bytes()
    if b"\x00" in raw:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return _notebook_to_text(text) if path.suffix.lower() == ".ipynb" else text


def _notebook_to_text(raw_json: str) -> str | None:
    """Keep notebook cell sources and drop the JSON, outputs and images.

    Line numbers for notebooks refer to this extracted text, not the .ipynb file.
    """
    try:
        cells = json.loads(raw_json).get("cells", [])
    except (json.JSONDecodeError, AttributeError):
        return None
    parts = []
    for cell in cells:
        source = cell.get("source", "")
        body = "".join(source) if isinstance(source, list) else str(source)
        parts.append(f"# %% [{cell.get('cell_type', 'cell')}]\n{body}")
    return "\n\n".join(parts)
