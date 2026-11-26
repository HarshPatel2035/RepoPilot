"""Command line: python -m repopilot.evals {retrieval,generation} --help"""

import argparse
import json
import logging
import sys
from pathlib import Path

from repopilot.config import Settings
from repopilot.embeddings import FastEmbedder
from repopilot.evals.dataset import load_eval_set
from repopilot.evals.judge import GroqJudge
from repopilot.evals.runner import (
    ensure_indexed,
    evaluate_generation,
    evaluate_retrieval,
    generation_to_markdown,
    report_to_markdown,
)
from repopilot.llm import GroqAnswerer
from repopilot.retrieval import FastReranker, parse_mode
from repopilot.store import Store

DEFAULT_DATASET = Path("evals/itsdangerous.json")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings()
    store = Store(settings.database_url, settings.embed_dim)
    store.init_schema()
    embedder = FastEmbedder(settings.embed_model)
    eval_set = load_eval_set(args.dataset)
    repo_id = ensure_indexed(store, embedder, settings, eval_set, reindex=args.reindex)
    reranker = None

    def get_reranker() -> FastReranker:
        nonlocal reranker
        reranker = reranker or FastReranker(settings.rerank_model)
        return reranker

    if args.command == "retrieval":
        modes = [parse_mode(m) for m in args.modes.split(",")]
        reports = [
            evaluate_retrieval(
                store, embedder, repo_id, eval_set, args.k, base, get_reranker() if rr else None
            )
            for base, rr in modes
        ]
        print(report_to_markdown(reports))
        _write(args.out, [r.to_dict() for r in reports])
        return _gate(reports, args)

    if not settings.groq_api_key:
        print("GROQ_API_KEY is not set; generation evals need it.", file=sys.stderr)
        return 2
    base, rr = parse_mode(args.mode)
    report = evaluate_generation(
        store,
        embedder,
        repo_id,
        eval_set,
        args.k,
        answerer=GroqAnswerer(settings.groq_api_key, settings.llm_model),
        judge=GroqJudge(settings.groq_api_key, settings.judge_model),
        mode=base,
        reranker=get_reranker() if rr else None,
    )
    print(generation_to_markdown(report))
    _write(args.out, report.to_dict())
    return 0


def _gate(reports: list, args: argparse.Namespace) -> int:
    """Fail (exit 1) when the gated mode drops below the thresholds, so CI blocks the change."""
    gated = next((r for r in reports if r.mode == args.gate_mode), None)
    if gated is None:
        print(f"gate mode {args.gate_mode!r} was not evaluated", file=sys.stderr)
        return 2
    failures = [
        f"{name} {value:.3f} < {minimum:.3f}"
        for name, value, minimum in [
            ("hit rate", gated.hit_rate, args.min_hit_rate),
            ("MRR", gated.mrr, args.min_mrr),
        ]
        if value < minimum
    ]
    if failures:
        print(f"Retrieval quality gate FAILED for {gated.mode}: " + "; ".join(failures))
        return 1
    print(f"Retrieval quality gate passed for {gated.mode}.")
    return 0


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m repopilot.evals")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--k", type=int, default=8, help="chunks per question (API default: 8)")
    parser.add_argument("--reindex", action="store_true", help="rebuild the eval index")
    sub = parser.add_subparsers(dest="command", required=True)

    retrieval = sub.add_parser("retrieval", help="score retrieval, no API key needed")
    retrieval.add_argument("--modes", default="vector,keyword,hybrid,hybrid+rerank")
    retrieval.add_argument("--gate-mode", default="hybrid")
    retrieval.add_argument("--min-hit-rate", type=float, default=0.0)
    retrieval.add_argument("--min-mrr", type=float, default=0.0)
    retrieval.add_argument("--out", type=Path, default=Path("evals/results/retrieval.json"))

    generation = sub.add_parser("generation", help="answer + LLM judge, needs GROQ_API_KEY")
    generation.add_argument("--mode", default="hybrid")
    generation.add_argument("--out", type=Path, default=Path("evals/results/generation.json"))
    return parser


if __name__ == "__main__":
    sys.exit(main())
