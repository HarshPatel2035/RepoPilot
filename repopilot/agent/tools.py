"""Read-only tools the agent can call.

Every tool records what it showed the model in `evidence`, so answers can be traced to
sources and evaluated. There is deliberately no tool that runs repository code: executing
code from an arbitrary public repo needs a proper sandbox first.
"""

from typing import Any, Protocol

from langchain_core.tools import BaseTool, StructuredTool

from repopilot.embeddings import Embedder
from repopilot.retrieval import Searcher, retrieve

SEARCH_K = 6
MAX_SNIPPET_CHARS = 1_500
MAX_READ_LINES = 150
MAX_LISTED_PATHS = 200


class FileStore(Searcher, Protocol):
    def list_paths(self, repo_id: int, prefix: str = "") -> list[str]: ...

    def read_file(self, repo_id: int, path: str) -> str | None: ...


def _label(path: str, start: int, end: int) -> str:
    return f"{path}:{start}-{end}"


def make_tools(
    store: FileStore, embedder: Embedder, repo_id: int, evidence: list[dict[str, Any]]
) -> list[BaseTool]:
    """Build the tools for one agent run; `evidence` collects everything they return."""

    def search_code(query: str) -> str:
        """Search the repository for code or docs about `query`.

        Returns the best matching chunks, each labelled [path:start-end].
        """
        hits = retrieve(store, embedder, repo_id, query, SEARCH_K)
        if not hits:
            return "No matches."
        evidence.extend(
            {k: h[k] for k in ("path", "start_line", "end_line", "content")} for h in hits
        )
        return "\n\n".join(
            f"[{_label(h['path'], h['start_line'], h['end_line'])}]\n"
            f"{h['content'][:MAX_SNIPPET_CHARS]}"
            for h in hits
        )

    def list_files(prefix: str = "") -> str:
        """List file paths in the repository, optionally only those starting with `prefix`
        (for example 'src/')."""
        paths = store.list_paths(repo_id, prefix)
        if not paths:
            return f"No files start with {prefix!r}."
        shown = paths[:MAX_LISTED_PATHS]
        more = len(paths) - len(shown)
        return "\n".join(shown) + (f"\n... {more} more; use a longer prefix" if more else "")

    def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read a file's lines, numbered. Returns at most 150 lines per call; use
        start_line and end_line to read long files in parts."""
        text = store.read_file(repo_id, path)
        if text is None:
            return f"File {path!r} not found. Use list_files to see the available paths."
        lines = text.splitlines()
        start = max(start_line, 1)
        if start > len(lines):
            return f"start_line {start} is past the end of {path} ({len(lines)} lines)."
        wanted_end = len(lines) if end_line is None else min(end_line, len(lines))
        if wanted_end < start:
            return "end_line must be greater than or equal to start_line."
        end = min(wanted_end, start + MAX_READ_LINES - 1)
        evidence.append(
            {
                "path": path,
                "start_line": start,
                "end_line": end,
                "content": "\n".join(lines[start - 1 : end]),
            }
        )
        numbered = "\n".join(f"{n}: {lines[n - 1]}" for n in range(start, end + 1))
        note = (
            f"\n[truncated: showed lines {start}-{end} of {len(lines)}; "
            f"call again with start_line={end + 1}]"
            if end < wanted_end
            else ""
        )
        return f"[{_label(path, start, end)}]\n{numbered}{note}"

    return [StructuredTool.from_function(f) for f in (search_code, list_files, read_file)]
