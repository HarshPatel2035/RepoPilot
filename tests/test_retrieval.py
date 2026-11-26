import numpy as np
import pytest

from repopilot.retrieval import parse_mode, retrieve
from tests.conftest import HashEmbedder


class RecordingStore:
    def __init__(self, hits: list[dict]) -> None:
        self.hits = hits
        self.calls: list[dict] = []

    def search(
        self, repo_id: int, query_text: str, query_vector: np.ndarray, k: int, mode: str
    ) -> list[dict]:
        self.calls.append({"k": k, "mode": mode})
        return self.hits[:k]


class LengthReranker:
    """Scores longer content higher, so the expected order is easy to see."""

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [float(len(d)) for d in documents]


def _hit(n: int, content: str) -> dict:
    return {"path": f"f{n}.py", "start_line": 1, "end_line": 2, "content": content, "score": 0.1}


HITS = [_hit(1, "a"), _hit(2, "aaa"), _hit(3, "aa"), _hit(4, "aaaa")]


def test_retrieve_without_reranker_passes_k_and_mode_through() -> None:
    store = RecordingStore(HITS)

    hits = retrieve(store, HashEmbedder(), 1, "q", k=2, mode="vector")

    assert [h["path"] for h in hits] == ["f1.py", "f2.py"]
    assert store.calls == [{"k": 2, "mode": "vector"}]


def test_retrieve_with_reranker_reorders_a_larger_pool() -> None:
    store = RecordingStore(HITS)

    hits = retrieve(store, HashEmbedder(), 1, "q", k=2, mode="hybrid", reranker=LengthReranker())

    assert [h["path"] for h in hits] == ["f4.py", "f2.py"]
    assert hits[0]["score"] == float(len("f4.py\naaaa"))  # reranker sees path + content
    assert store.calls[0]["k"] > 2


def test_retrieve_with_reranker_and_no_hits() -> None:
    hits = retrieve(RecordingStore([]), HashEmbedder(), 1, "q", k=2, reranker=LengthReranker())

    assert hits == []


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("hybrid", ("hybrid", False)),
        ("vector", ("vector", False)),
        ("hybrid+rerank", ("hybrid", True)),
    ],
)
def test_parse_mode(name: str, expected: tuple[str, bool]) -> None:
    assert parse_mode(name) == expected


@pytest.mark.parametrize("name", ["bm25", "hybrid+magic", ""])
def test_parse_mode_rejects_unknown(name: str) -> None:
    with pytest.raises(ValueError):
        parse_mode(name)
