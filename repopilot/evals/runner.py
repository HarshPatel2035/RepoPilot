"""Run a golden set against the real pipelines (retrieval, RAG, agent) and score them."""

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Any, Protocol

from repopilot import indexer
from repopilot.agent.graph import RepoAgent
from repopilot.config import Settings
from repopilot.embeddings import Embedder
from repopilot.evals.dataset import EvalSet
from repopilot.evals.judge import Judgement
from repopilot.evals.metrics import citation_precision, hit_rate, ndcg, recall, reciprocal_rank
from repopilot.llm import Answerer
from repopilot.retrieval import Reranker, retrieve
from repopilot.store import Store

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalReport:
    mode: str
    k: int
    hit_rate: float
    recall: float
    mrr: float
    ndcg: float
    p50_latency_ms: float  # median: robust to the first, cold call
    rows: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationReport:
    pipeline: str  # "rag" or "agent"
    judged: int
    faithfulness: float
    correctness: float
    citation_precision: float | None  # mean over answers that cite anything
    cited_share: float  # share of answers with at least one citation
    evidence_hit_rate: float  # share of questions where the model was shown a gold span
    rows: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Judge(Protocol):
    def judge(
        self, question: str, hits: list[dict[str, Any]], answer: str, reference: str
    ) -> Judgement: ...


# A pipeline answers one question and returns (answer, evidence it was shown).
Pipeline = Callable[[str], tuple[str, list[dict[str, Any]]]]


def ensure_indexed(
    store: Store, embedder: Embedder, settings: Settings, eval_set: EvalSet, reindex: bool = False
) -> int:
    """Index the pinned repo once; later runs reuse it. Raises if indexing fails."""
    key = f"{eval_set.repo}@{eval_set.ref}"  # separate from any HEAD copy indexed via the API
    repo = store.upsert_repo(key)
    if repo["status"] == "ready" and not reindex:
        return repo["id"]
    indexer.index_repo(store, embedder, settings, repo["id"], eval_set.repo, ref=eval_set.ref)
    indexed = store.get_repo(repo["id"])
    if indexed is None or indexed["status"] != "ready":
        raise RuntimeError(f"indexing {key} failed: {indexed and indexed['error']}")
    return repo["id"]


def _mode_name(mode: str, reranker: Reranker | None) -> str:
    return f"{mode}+rerank" if reranker else mode


def _label(hit: dict[str, Any]) -> str:
    return f"{hit['path']}:{hit['start_line']}-{hit['end_line']}"


def evaluate_retrieval(
    store: Store,
    embedder: Embedder,
    repo_id: int,
    eval_set: EvalSet,
    k: int,
    mode: str = "hybrid",
    reranker: Reranker | None = None,
) -> RetrievalReport:
    rows = []
    for q in eval_set.questions:
        started = time.perf_counter()
        hits = retrieve(store, embedder, repo_id, q.question, k, mode=mode, reranker=reranker)
        latency_ms = (time.perf_counter() - started) * 1000
        rows.append(
            {
                "id": q.id,
                "latency_ms": round(latency_ms, 1),
                "hit": hit_rate(hits, q.relevant),
                "recall": recall(hits, q.relevant),
                "rr": reciprocal_rank(hits, q.relevant),
                "ndcg": ndcg(hits, q.relevant),
                "retrieved": [_label(h) for h in hits],
            }
        )
    return RetrievalReport(
        mode=_mode_name(mode, reranker),
        k=k,
        hit_rate=mean(r["hit"] for r in rows),
        recall=mean(r["recall"] for r in rows),
        mrr=mean(r["rr"] for r in rows),
        ndcg=mean(r["ndcg"] for r in rows),
        p50_latency_ms=median(r["latency_ms"] for r in rows),
        rows=tuple(rows),
    )


def rag_pipeline(
    store: Store,
    embedder: Embedder,
    repo_id: int,
    k: int,
    answerer: Answerer,
    mode: str = "hybrid",
    reranker: Reranker | None = None,
) -> Pipeline:
    """One retrieval, one LLM call: what POST /ask does."""

    def run(question: str) -> tuple[str, list[dict[str, Any]]]:
        hits = retrieve(store, embedder, repo_id, question, k, mode=mode, reranker=reranker)
        return answerer.answer(question, hits), hits

    return run


def agent_pipeline(agent: RepoAgent, repo_id: int) -> Pipeline:
    """Multi-step tool use: what POST /agent does."""

    def run(question: str) -> tuple[str, list[dict[str, Any]]]:
        result = agent.run(repo_id, question)
        return result.answer, list(result.evidence)

    return run


def evaluate_generation(
    eval_set: EvalSet, pipeline: Pipeline, judge: Judge, name: str
) -> GenerationReport:
    rows = []
    for q in eval_set.questions:
        answer, evidence = pipeline(q.question)
        row: dict[str, Any] = {
            "id": q.id,
            "answer": answer,
            "evidence": [_label(e) for e in evidence],
            "evidence_hit": hit_rate(evidence, q.relevant),
            "citation_precision": citation_precision(answer, evidence),
        }
        try:
            row.update(asdict(judge.judge(q.question, evidence, answer, q.reference)))
        except ValueError as exc:  # malformed judge output: keep the row, skip its scores
            log.warning("judge failed on %s: %s", q.id, exc)
            row["judge_error"] = str(exc)
        rows.append(row)

    judged = [r for r in rows if "faithfulness" in r]
    cited = [r["citation_precision"] for r in rows if r["citation_precision"] is not None]
    return GenerationReport(
        pipeline=name,
        judged=len(judged),
        faithfulness=mean(r["faithfulness"] for r in judged) if judged else 0.0,
        correctness=mean(r["correctness"] for r in judged) if judged else 0.0,
        citation_precision=mean(cited) if cited else None,
        cited_share=len(cited) / len(rows) if rows else 0.0,
        evidence_hit_rate=mean(r["evidence_hit"] for r in rows) if rows else 0.0,
        rows=tuple(rows),
    )


def report_to_markdown(reports: list[RetrievalReport]) -> str:
    lines = [
        "| Mode | k | Hit rate | Recall | MRR | nDCG | p50 latency |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r.mode} | {r.k} | {r.hit_rate:.3f} | {r.recall:.3f} | {r.mrr:.3f} "
        f"| {r.ndcg:.3f} | {r.p50_latency_ms:.0f} ms |"
        for r in reports
    ]
    return "\n".join(lines)


def generation_to_markdown(reports: list[GenerationReport]) -> str:
    lines = [
        "| Pipeline | Judged | Faithfulness (1-5) | Correctness (1-5) | Citation precision "
        "| Answers citing | Evidence hit rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        precision = "n/a" if r.citation_precision is None else f"{r.citation_precision:.3f}"
        lines.append(
            f"| {r.pipeline} | {r.judged} | {r.faithfulness:.2f} | {r.correctness:.2f} "
            f"| {precision} | {r.cited_share:.0%} | {r.evidence_hit_rate:.3f} |"
        )
    return "\n".join(lines)
