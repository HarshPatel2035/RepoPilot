"""Run a golden set against the real retrieval pipeline and summarise the scores."""

import logging
import time
from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Any

from repopilot import indexer
from repopilot.config import Settings
from repopilot.embeddings import Embedder
from repopilot.evals.dataset import EvalSet
from repopilot.evals.judge import GroqJudge, Judgement
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
    mode: str
    k: int
    judged: int
    faithfulness: float
    correctness: float
    citation_precision: float | None  # mean over answers that cite anything
    cited_share: float  # share of answers with at least one citation
    rows: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def evaluate_generation(
    store: Store,
    embedder: Embedder,
    repo_id: int,
    eval_set: EvalSet,
    k: int,
    answerer: Answerer,
    judge: GroqJudge,
    mode: str = "hybrid",
    reranker: Reranker | None = None,
) -> GenerationReport:
    rows = []
    for q in eval_set.questions:
        hits = retrieve(store, embedder, repo_id, q.question, k, mode=mode, reranker=reranker)
        answer = answerer.answer(q.question, hits)
        row: dict[str, Any] = {
            "id": q.id,
            "answer": answer,
            "citation_precision": citation_precision(answer, hits),
        }
        try:
            verdict: Judgement = judge.judge(q.question, hits, answer, q.reference)
            row.update(asdict(verdict))
        except ValueError as exc:  # malformed judge output: keep the row, skip its scores
            log.warning("judge failed on %s: %s", q.id, exc)
            row["judge_error"] = str(exc)
        rows.append(row)

    judged = [r for r in rows if "faithfulness" in r]
    cited = [r["citation_precision"] for r in rows if r["citation_precision"] is not None]
    return GenerationReport(
        mode=_mode_name(mode, reranker),
        k=k,
        judged=len(judged),
        faithfulness=mean(r["faithfulness"] for r in judged) if judged else 0.0,
        correctness=mean(r["correctness"] for r in judged) if judged else 0.0,
        citation_precision=mean(cited) if cited else None,
        cited_share=len(cited) / len(rows) if rows else 0.0,
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


def generation_to_markdown(report: GenerationReport) -> str:
    precision = "n/a" if report.citation_precision is None else f"{report.citation_precision:.3f}"
    return "\n".join(
        [
            "| Mode | k | Judged | Faithfulness (1-5) | Correctness (1-5) | Citation precision "
            "| Answers citing |",
            "|---|---|---|---|---|---|---|",
            f"| {report.mode} | {report.k} | {report.judged} | {report.faithfulness:.2f} "
            f"| {report.correctness:.2f} | {precision} | {report.cited_share:.0%} |",
        ]
    )
