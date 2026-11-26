import math

import pytest

from repopilot.evals.metrics import (
    Span,
    citation_precision,
    hit_rate,
    ndcg,
    novel_relevance,
    parse_citations,
    recall,
    reciprocal_rank,
)

TARGET = Span("src/a.py", 50, 60)
OTHER = Span("docs/b.rst", 1, 10)


def _hit(path: str, start: int, end: int) -> dict:
    return {"path": path, "start_line": start, "end_line": end}


MISS = _hit("src/z.py", 1, 40)
ON_TARGET = _hit("src/a.py", 41, 80)
OVERLAPPING_TARGET_AGAIN = _hit("src/a.py", 31, 70)
ON_OTHER = _hit("docs/b.rst", 1, 40)


def test_novel_relevance_counts_each_gold_span_once() -> None:
    hits = [MISS, ON_TARGET, OVERLAPPING_TARGET_AGAIN, ON_OTHER]

    assert novel_relevance(hits, [TARGET, OTHER]) == [False, True, False, True]


def test_same_path_without_line_overlap_is_not_relevant() -> None:
    assert novel_relevance([_hit("src/a.py", 1, 40)], [TARGET]) == [False]


def test_hit_rate_recall_and_reciprocal_rank() -> None:
    hits = [MISS, ON_TARGET, MISS]

    assert hit_rate(hits, [TARGET, OTHER]) == 1.0
    assert recall(hits, [TARGET, OTHER]) == 0.5
    assert reciprocal_rank(hits, [TARGET, OTHER]) == 0.5
    assert hit_rate([MISS], [TARGET]) == 0.0
    assert reciprocal_rank([MISS], [TARGET]) == 0.0


def test_ndcg_is_one_for_perfect_ranking_and_lower_when_pushed_down() -> None:
    perfect = ndcg([ON_TARGET, ON_OTHER, MISS], [TARGET, OTHER])
    pushed_down = ndcg([MISS, ON_TARGET, ON_OTHER], [TARGET, OTHER])
    # DCG = 1/log2(3) + 1/log2(4); ideal DCG = 1/log2(2) + 1/log2(3)
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))

    assert perfect == pytest.approx(1.0)
    assert pushed_down == pytest.approx(expected)


def test_ndcg_never_exceeds_one_when_chunks_overlap() -> None:
    assert ndcg([ON_TARGET, OVERLAPPING_TARGET_AGAIN], [TARGET]) == pytest.approx(1.0)


def test_metrics_on_empty_results_are_zero() -> None:
    assert hit_rate([], [TARGET]) == 0.0
    assert recall([], [TARGET]) == 0.0
    assert reciprocal_rank([], [TARGET]) == 0.0
    assert ndcg([], [TARGET]) == 0.0


def test_parse_citations_reads_path_and_line_range() -> None:
    answer = "Uses compare_digest [src/a.py:24-28] and docs [docs/b.rst:1-10]. Not [this]."

    assert parse_citations(answer) == [Span("src/a.py", 24, 28), Span("docs/b.rst", 1, 10)]


def test_citation_precision_is_share_of_citations_backed_by_retrieved_chunks() -> None:
    answer = "Real [src/a.py:50-55], invented [src/made_up.py:1-5]."

    assert citation_precision(answer, [ON_TARGET]) == 0.5
    assert citation_precision("no citations at all", [ON_TARGET]) is None
