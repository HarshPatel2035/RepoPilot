"""Retrieval pipeline shared by the API and the evaluations: search, then optionally rerank."""

from typing import Any, Protocol

import numpy as np

from repopilot.embeddings import Embedder
from repopilot.store import SEARCH_MODES

# Rerank a pool this many times bigger than k: big enough to rescue good chunks the first
# stage ranked low, small enough to keep the cross-encoder fast on CPU.
RERANK_POOL_FACTOR = 4


class Searcher(Protocol):
    def search(
        self, repo_id: int, query_text: str, query_vector: np.ndarray, k: int, mode: str
    ) -> list[dict[str, Any]]: ...


class Reranker(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]: ...


class FastReranker:
    """Cross-encoder reranker (ONNX via fastembed): reads query and chunk together."""

    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name)

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, documents)]


def parse_mode(name: str) -> tuple[str, bool]:
    """'hybrid+rerank' -> ('hybrid', True); 'vector' -> ('vector', False)."""
    base, plus, extra = name.partition("+")
    if base not in SEARCH_MODES or (plus and extra != "rerank"):
        raise ValueError(f"unknown retrieval mode {name!r}")
    return base, bool(plus)


def retrieve(
    store: Searcher,
    embedder: Embedder,
    repo_id: int,
    question: str,
    k: int,
    mode: str = "hybrid",
    reranker: Reranker | None = None,
) -> list[dict[str, Any]]:
    pool = k * RERANK_POOL_FACTOR if reranker else k
    hits = store.search(repo_id, question, embedder.embed_query(question), pool, mode=mode)
    if reranker is None or not hits:
        return hits
    documents = [f"{h['path']}\n{h['content']}" for h in hits]
    scored = zip(reranker.score(question, documents), hits, strict=True)
    ranked = sorted(scored, key=lambda pair: pair[0], reverse=True)
    return [{**hit, "score": score} for score, hit in ranked[:k]]
