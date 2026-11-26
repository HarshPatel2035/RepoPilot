"""Postgres + pgvector storage for repositories and their chunks, with hybrid search."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from repopilot.ingest import Chunk

# Reciprocal Rank Fusion constant: 60 is the value from the original RRF paper.
RRF_K = 60
MIN_CANDIDATES = 20
SEARCH_MODES = frozenset({"hybrid", "vector", "keyword"})

_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS repos (
    id          BIGSERIAL PRIMARY KEY,
    url         TEXT UNIQUE NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    error       TEXT,
    chunk_count INT NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS chunks (
    id         BIGSERIAL PRIMARY KEY,
    repo_id    BIGINT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    path       TEXT NOT NULL,
    start_line INT NOT NULL,
    end_line   INT NOT NULL,
    content    TEXT NOT NULL,
    embedding  vector({dim}) NOT NULL,
    tsv        tsvector GENERATED ALWAYS AS (to_tsvector('english', path || ' ' || content)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_repo_idx ON chunks (repo_id);
CREATE TABLE IF NOT EXISTS files (
    repo_id BIGINT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    path    TEXT NOT NULL,
    content TEXT NOT NULL,
    PRIMARY KEY (repo_id, path)
);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
"""
# 'english' text search drops stopwords ("how", "the") and stems words; the eval showed it
# raises keyword hit@5 from 0.42 to 0.69 over 'simple' (see evals/README.md).
# shortcut: no migrations; CREATE IF NOT EXISTS won't alter an existing table, so drop the
# tables after schema changes. Add Alembic or similar before real users have data.
# shortcut: exact vector scan per repo (fine up to ~100k chunks); add an HNSW index with
# iterative scans when repos get bigger.

_SEARCH = """
WITH vec AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> %(qvec)s) AS rank
    FROM chunks
    WHERE repo_id = %(repo_id)s
    ORDER BY embedding <=> %(qvec)s
    LIMIT %(vec_limit)s
),
kw AS (
    SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS rank
    FROM chunks, to_tsquery('english', %(tsquery)s) AS q
    WHERE repo_id = %(repo_id)s AND tsv @@ q
    ORDER BY ts_rank_cd(tsv, q) DESC
    LIMIT %(kw_limit)s
)
SELECT c.id, c.path, c.start_line, c.end_line, c.content,
       (COALESCE(1.0 / (%(rrf_k)s + vec.rank), 0)
        + COALESCE(1.0 / (%(rrf_k)s + kw.rank), 0))::float8 AS score
FROM vec
FULL OUTER JOIN kw ON vec.id = kw.id
JOIN chunks c ON c.id = COALESCE(vec.id, kw.id)
ORDER BY score DESC, c.id
LIMIT %(k)s
"""


class Store:
    def __init__(self, dsn: str, embed_dim: int) -> None:
        self._dsn = dsn
        self._dim = int(embed_dim)

    @contextmanager
    def _connect(self) -> Iterator[psycopg.Connection[dict[str, Any]]]:
        # shortcut: one connection per call; switch to psycopg_pool when traffic grows.
        with psycopg.connect(self._dsn, row_factory=dict_row) as conn:
            register_vector(conn)
            yield conn

    def init_schema(self) -> None:
        # Plain connection: register_vector needs the extension this creates.
        with psycopg.connect(self._dsn) as conn:
            conn.execute(_SCHEMA.format(dim=self._dim))

    def ping(self) -> None:
        with psycopg.connect(self._dsn, connect_timeout=3) as conn:
            conn.execute("SELECT 1")

    def upsert_repo(self, url: str) -> dict[str, Any]:
        """Insert the repo if new. The returned row has `inserted` = True only on first insert."""
        with self._connect() as conn:
            return conn.execute(
                """
                INSERT INTO repos (url) VALUES (%s)
                ON CONFLICT (url) DO UPDATE SET url = EXCLUDED.url
                RETURNING *, (xmax = 0) AS inserted
                """,
                (url,),
            ).fetchone()

    def get_repo(self, repo_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            return conn.execute("SELECT * FROM repos WHERE id = %s", (repo_id,)).fetchone()

    def list_repos(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return conn.execute("SELECT * FROM repos ORDER BY created_at DESC, id DESC").fetchall()

    def set_status(
        self, repo_id: int, status: str, *, error: str | None = None, chunk_count: int | None = None
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE repos
                SET status = %s, error = %s, chunk_count = COALESCE(%s, chunk_count),
                    updated_at = now()
                WHERE id = %s
                """,
                (status, error, chunk_count, repo_id),
            )

    def replace_chunks(
        self,
        repo_id: int,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        files: Sequence[tuple[str, str]] = (),
    ) -> None:
        """Atomically swap a repo's chunks and files, so readers never see a half-built index.

        `files` are (path, full text) pairs; the agent's read_file tool reads them.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        rows = [
            (repo_id, c.path, c.start_line, c.end_line, c.content, np.asarray(v, dtype=np.float32))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        with self._connect() as conn, conn.transaction(), conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE repo_id = %s", (repo_id,))
            cur.execute("DELETE FROM files WHERE repo_id = %s", (repo_id,))
            cur.executemany(
                """
                INSERT INTO chunks (repo_id, path, start_line, end_line, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                rows,
            )
            cur.executemany(
                "INSERT INTO files (repo_id, path, content) VALUES (%s, %s, %s)",
                [(repo_id, path, text) for path, text in files],
            )

    def list_paths(self, repo_id: int, prefix: str = "") -> list[str]:
        # starts_with() instead of LIKE, so "_" and "%" in a prefix are not wildcards.
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT path FROM files WHERE repo_id = %s AND starts_with(path, %s) ORDER BY path",
                (repo_id, prefix),
            ).fetchall()
        return [r["path"] for r in rows]

    def read_file(self, repo_id: int, path: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content FROM files WHERE repo_id = %s AND path = %s", (repo_id, path)
            ).fetchone()
        return row["content"] if row else None

    def search(
        self,
        repo_id: int,
        query_text: str,
        query_vector: Sequence[float],
        k: int,
        mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """Top-k chunks by vector similarity, keyword match, or both merged with RRF.

        "vector" and "keyword" exist so evaluations can measure what each half adds.
        """
        if mode not in SEARCH_MODES:
            raise ValueError(f"unknown search mode {mode!r}")
        candidates = max(k * 4, MIN_CANDIDATES)
        params = {
            "repo_id": repo_id,
            "qvec": np.asarray(query_vector, dtype=np.float32),
            "tsquery": _to_or_tsquery(query_text),
            "vec_limit": 0 if mode == "keyword" else candidates,
            "kw_limit": 0 if mode == "vector" else candidates,
            "rrf_k": RRF_K,
            "k": k,
        }
        with self._connect() as conn:
            return conn.execute(_SEARCH, params).fetchall()


def _to_or_tsquery(text: str) -> str:
    """'How does LRUCache evict?' -> 'does | evict | how | lrucache'.

    OR-ing words means a chunk only needs some of the question's words to match.
    Only \\w characters survive, so user text can never break tsquery syntax.
    """
    words = sorted({w.lower() for w in re.findall(r"\w+", text) if len(w) > 2})
    return " | ".join(words)
