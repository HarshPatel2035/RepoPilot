"""Clone -> filter -> chunk -> embed -> store, for one repository."""

import logging
import tempfile
from pathlib import Path

from repopilot.config import Settings
from repopilot.embeddings import Embedder
from repopilot.ingest import Chunk, IngestError, chunk_lines, clone_repo, iter_source_files
from repopilot.store import Store

log = logging.getLogger(__name__)


def embedding_text(chunk: Chunk) -> str:
    """Prefix the path so a question about 'auth' also matches code in auth.py."""
    return f"{chunk.path}\n{chunk.content}"


def index_repo(
    store: Store,
    embedder: Embedder,
    settings: Settings,
    repo_id: int,
    url: str,
    ref: str | None = None,
) -> None:
    """Build a repo's index. Never raises: failures are recorded on the repo row."""
    store.set_status(repo_id, "indexing")
    try:
        with tempfile.TemporaryDirectory(prefix="repopilot-") as tmp:
            checkout = Path(tmp) / "repo"
            clone_repo(url, checkout, ref=ref)
            files = iter_source_files(checkout, settings.max_file_bytes, settings.max_repo_files)
            chunks = [chunk for path, text in files for chunk in chunk_lines(path, text)]
        if not chunks:
            raise IngestError("no supported source files found in the repository")
        vectors = embedder.embed_documents([embedding_text(c) for c in chunks])
        store.replace_chunks(repo_id, chunks, vectors)
        store.set_status(repo_id, "ready", chunk_count=len(chunks))
        log.info("indexed repo %s (%s): %d chunks", repo_id, url, len(chunks))
    except IngestError as exc:
        log.warning("indexing repo %s failed: %s", repo_id, exc)
        store.set_status(repo_id, "failed", error=str(exc))
    except Exception:
        log.exception("unexpected error indexing repo %s", repo_id)
        store.set_status(
            repo_id, "failed", error="unexpected error while indexing; see server logs"
        )
