import os
import re
import zlib

import numpy as np
import psycopg
import pytest

from repopilot.store import Store

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")
EMBED_DIM = 384

requires_db = pytest.mark.skipif(
    not TEST_DATABASE_URL, reason="set TEST_DATABASE_URL to run database tests"
)


class HashEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words get similar vectors."""

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"\w+", text.lower()):
            vector[zlib.crc32(token.encode()) % self.dim] += 1.0
        if not vector.any():
            vector[0] = 1.0  # cosine distance is undefined for a zero vector
        return vector / np.linalg.norm(vector)

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)


class FakeAnswerer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict]]] = []

    def answer(self, question: str, hits: list[dict]) -> str:
        self.calls.append((question, hits))
        return "fake answer"


@pytest.fixture
def store() -> Store:
    if not TEST_DATABASE_URL:
        pytest.skip("set TEST_DATABASE_URL to run database tests")
    db = Store(TEST_DATABASE_URL, EMBED_DIM)
    db.init_schema()
    with psycopg.connect(TEST_DATABASE_URL) as conn:
        conn.execute("TRUNCATE repos RESTART IDENTITY CASCADE")
    return db


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder()
