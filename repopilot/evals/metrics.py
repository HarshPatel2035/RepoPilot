"""Retrieval and citation metrics. Pure functions, so they are easy to test and explain.

A retrieved chunk is relevant when it is in the same file as a gold span and their line
ranges overlap. Each gold span counts once: a second chunk overlapping an already-found span
adds nothing, which keeps nDCG <= 1 even though chunks overlap each other.
"""

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

Hit = Mapping[str, Any]

_CITATION = re.compile(r"\[([^\[\]:\s]+):(\d+)-(\d+)\]")


@dataclass(frozen=True)
class Span:
    path: str
    start: int  # 1-based, inclusive
    end: int  # 1-based, inclusive


def _overlaps(hit: Hit, span: Span) -> bool:
    return (
        hit["path"] == span.path and hit["start_line"] <= span.end and hit["end_line"] >= span.start
    )


def novel_relevance(hits: Sequence[Hit], spans: Sequence[Span]) -> list[bool]:
    """Per hit: does it cover a gold span that no higher-ranked hit covered?"""
    found: set[Span] = set()
    flags = []
    for hit in hits:
        new = {s for s in spans if s not in found and _overlaps(hit, s)}
        found |= new
        flags.append(bool(new))
    return flags


def hit_rate(hits: Sequence[Hit], spans: Sequence[Span]) -> float:
    """1.0 if any gold span was retrieved, else 0.0 (averaged over questions = hit rate @k)."""
    return 1.0 if any(novel_relevance(hits, spans)) else 0.0


def recall(hits: Sequence[Hit], spans: Sequence[Span]) -> float:
    """Share of gold spans covered by the retrieved chunks."""
    return sum(novel_relevance(hits, spans)) / len(spans)


def reciprocal_rank(hits: Sequence[Hit], spans: Sequence[Span]) -> float:
    """1/rank of the first relevant chunk (averaged over questions = MRR)."""
    for rank, relevant in enumerate(novel_relevance(hits, spans), start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def ndcg(hits: Sequence[Hit], spans: Sequence[Span]) -> float:
    """Normalised discounted cumulative gain with binary relevance."""
    flags = novel_relevance(hits, spans)
    dcg = sum(1.0 / math.log2(rank + 1) for rank, rel in enumerate(flags, start=1) if rel)
    ideal_hits = min(len(spans), len(hits))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal if ideal else 0.0


def parse_citations(answer: str) -> list[Span]:
    """Extract [path:start-end] citations from an answer."""
    return [Span(p, int(a), int(b)) for p, a, b in _CITATION.findall(answer)]


def citation_precision(answer: str, hits: Sequence[Hit]) -> float | None:
    """Share of citations that point at a chunk the model was actually given.

    None when the answer has no citations (reported separately as `cited_share`).
    A citation outside the retrieved chunks means the model invented a source.
    """
    citations = parse_citations(answer)
    if not citations:
        return None
    backed = sum(any(_overlaps(h, c) for h in hits) for c in citations)
    return backed / len(citations)
